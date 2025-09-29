import numpy as np
from sequential.layers import Layer
from sequential.activations import Softmax


class Attention(Layer):
    '''
    Transformer multi-head self-attention layer.

    Projects sequential inputs into query (Q), key (K), 
    and value (V) representations (more details below) using 
    learned linear transformations. Computes scaled dot-product 
    attention between Q and K to produce attention weights, which are 
    then applied to V. Multiple attention heads are computed 
    in parallel, concatenated, and projected back to the model 
    dimension with a final linear output layer.

    Supports optional look-ahead masking to prevent future time 
    steps from contributing to the attention scores.

    Projected representations
    -------------------------
    Q, K, and V are all derived from the same input sequence (X) 
    but transformed into different representations through learned 
    projection weights. 

        Q (queries): information from each time step that searches 
        for relevant patterns in other time steps (K).

        K (keys): compared against the queries (Q) to determine
        relevance across all time steps. The optional look-ahead 
        mask blocks access to future time steps.

        V: multiplied by the attention scores between K and Q 
        to determine how much of each value contributes to the 
        final output.
    '''

    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.softmax = Softmax()
        self.built = False

    def __call__(self, X, mask=None):
        '''
        Args
        ----
        X: np.ndarray
            Inputs of shape (batch_size, time_steps, d_model).
        mask: np.ndarray or None
            optional look-ahead mask preventing future time steps 
            from being attended to, ensuring autoregressive consistency. 
            Shape is (1, 1, time_steps, time_steps) for multi-head attention 
            or (1, time_steps, time_steps) if num_heads == 1. Mask values are
            added to the attention scores with a 0 for allowed positions and 
            inf for disallowed positions.
        '''
        if not self.built:
            self.build(X)
        # unpack trainable params
        W = self.trainable_params['W']
        Wo = self.trainable_params['Wo']
        # linear projection of the input
        X_proj = np.dot(X, W)
        # split projected X if multi-headed
        if self.num_heads > 1:
            X_proj = self.split_heads(X_proj, multiplier=3)
        # slice Q, K, V out of the last dimension
        dim_size = self.d_head if self.num_heads > 1 else self.d_model
        Q = X_proj[..., :dim_size]
        K = X_proj[..., dim_size:2*dim_size]
        V = X_proj[..., 2*dim_size:3*dim_size]
        # compute attention scores
        attn, soft_out = self.scaled_dot_product_attention(Q, K, V, mask=mask)
        # combine heads if multi-headed
        if self.num_heads > 1:
            attn = self.combine_heads(attn)
        # final output projection
        output = np.matmul(attn, Wo)
        # cache intermediate variables for backprop
        self.fcache = {'X': X, 'Q': Q, 'K': K, 'V': V, 'attn': attn, 'soft_out': soft_out}

        return output

    def backward(self, upstream_grad):
        '''
        Propagates gradients backwards through the layer.

        Variable names mirror those in the forward pass, with a
        leading 'd' to indicate derivatives. For example:

            Q -> dQ
            soft_out -> dsoft_out
            X -> dX

        This helps track how forward variables contribute to 
        the propagated gradients.
        '''
        # unpack trainable params
        W = self.trainable_params['W']
        Wo = self.trainable_params['Wo']
        # unpack intermediate variables
        X = self.fcache['X']
        Q = self.fcache['Q']
        K = self.fcache['K']
        V = self.fcache['V']
        attn = self.fcache['attn']
        soft_out = self.fcache['soft_out']
        # gradient of attn in output = np.matmul(attn, Wo)
        dattn = np.dot(upstream_grad, Wo.T)
        # gradient of Wo in output = np.matmul(attn, Wo)
        if upstream_grad.ndim == 3:
           # align attn and upstream_grad for matrix multiplication,
           # then sum over batches so that dWo.shape == Wo.shape
            dWo = np.sum(np.matmul(attn.transpose(0, 2, 1), upstream_grad), axis=0)
        else:
            dWo = np.dot(attn.T, upstream_grad)
        # if multi-headed, the gradient of attn needs to be split before
        # backproping into scaled_dot_product_attention()
        if self.num_heads > 1:
            dattn = self.split_heads(dattn)

        # ---- backprop into scaled_dot_product_attention() ----

        # gradient of soft_out in np.matmul(soft_out, V)
        V_T = V.transpose(0, 1, 3, 2) if self.num_heads > 1 else V.transpose(0, 2, 1)
        dsoft_out = np.matmul(dattn, V_T)
        # gradient of V in np.matmul(soft_out, V)
        soft_out_T = soft_out.transpose(
            0, 1, 3, 2) if self.num_heads > 1 else soft_out.transpose(0, 2, 1)
        dV = np.matmul(soft_out_T, dattn)
        # gradient of softmax function
        dsoftmax_ = self.softmax.backward(soft_out, dsoft_out)
        # use the same scaling factor as in forward pass for computing
        # the gradients of Q and K
        scaler = np.sqrt(Q.shape[-1])
        dsoftmax_T = dsoftmax_.transpose(
            0, 1, 3, 2) if self.num_heads > 1 else dsoftmax_.transpose(0, 2, 1)
        # gradient of K in scores = np.matmul(Q, K) / scale
        dK = np.matmul(dsoftmax_T, Q) / scaler
        # gradient of Q in scores = np.matmul(Q, K) / scale
        dQ = np.matmul(dsoftmax_T, K) / scaler
        # concatenate gradients of Q, K, V since
        # W contains the weights for all 3, maintaining
        # the same order as in the forward pass
        dX = np.concatenate([dQ, dK, dV], axis=-1)
        # combine heads if multi-headed, setting multiplier to
        # 3 since dX is a concatenation of dQ, dK, dV
        if self.num_heads > 1:
            dX = self.combine_heads(dX, multiplier=3)

        # ---- backprop linear projections ----

        # gradient of W in X_proj = np.dot(X, W), aligning
        # X and dX for matrix multiplication, then summing over the
        # batches so that dW.shape == W.shape
        dW = np.sum(np.matmul(X.transpose(0, 2, 1), dX), axis=0)
        # cache trainable param gradients for optimization step
        self.trainable_params_grad = {'W': dW, 'Wo': dWo}
        # return gradient of X in X_proj = np.dot(X, W)
        return np.dot(dX, W.T)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # transpose K to (..., d_head, time_steps) so that it
        # aligns with Q's (..., time_steps, d_head) for the dot
        # product across the last two dimensions.
        if self.num_heads > 1:
            K = K.transpose(0, 1, 3, 2)
        else:
            K = K.transpose(0, 2, 1)
        # calculate attention scores, which are scaled to prevent
        # outputs from growing too large
        scaler = np.sqrt(Q.shape[-1])
        # the dot product of Q and K is scaled by their dimensionality
        # to prevent the result from growing too large
        scores = np.matmul(Q, K) / scaler
        if mask is not None:
            scores += mask
        soft_out = self.softmax(scores)
        # compute attention scores with the values matrix and
        # softmax probabilities
        attn = np.matmul(soft_out, V)

        return attn, soft_out

    def build(self, x):
        # initialize weights based on last dimension of the input (d_model)
        self.d_model = x.shape[-1]
        # calc the dimensionality per head, note that the output
        # of each head is concatenated to return to the d_model
        # dimensionality
        self.d_head = self.d_model // self.num_heads if self.num_heads > 1 else None
        # weights for Q, K, and V linear projections are combined into
        # a single matrix (W)
        W = np.random.randn(self.d_model, self.d_model * 3) * .01
        # output weights
        Wo = np.random.randn(self.d_model, self.d_model) * .01
        self.trainable_params = {'W': W, 'Wo': Wo}
        self.built = True

    def split_heads(self, x, multiplier=1):
        '''
        Split the last dimension of x into (num_heads, d_head * multiplier),
        then transpose to group each head with its full sequence of time 
        steps

        Args
        ----
        x: np.ndarray
            Shape (batch_size, time_steps, d_model * multiplier).
        multiplier: int
            Factor to expand the d_head dimension. 
                - 1 for a single projection (Q, K, or V).
                - 3 when working on concatenated QKV.
        '''
        # reshape so each head has its own (time_steps, d_head) slice
        # of the original d_model dimension.
        x = x.reshape(x.shape[0], x.shape[1], self.num_heads, self.d_head * multiplier)
        # transpose to (batch_size, num_heads, time_steps, d_head * multiplier).
        # This groups each head with a full sequence of time steps,
        # so that attention can be computed independently within each head.
        return x.transpose(0, 2, 1, 3)

    def combine_heads(self, x, multiplier=1):
        '''
        Combines the heads to revert x back to its 
        original shape (batch_size, time_steps, d_model)

        Args
        ----
        x: np.ndarray
            Shape (batch_size, num_heads, time_steps, d_head * multiplier).
        multiplier: int
            Factor to expand the d_model dimension. 
                - 1 for a single projection (Q, K, or V).
                - 3 when working on concatenated QKV.
        '''
        # transpose to (batch_size, time_steps, num_heads, d_head * multiplier)
        # so that the heads can be combined upon reshaping
        x = x.transpose(0, 2, 1, 3)
        # reshape to the original dimensions (batch_size, time_steps, d_model * multiplier)
        return x.reshape(x.shape[0], x.shape[1], self.d_model * multiplier)
