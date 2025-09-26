from sequential.layers import Dropout, LayerNorm
from sequential.layers.transformer import TransformerDense, Attention


class DecoderLayer:
    '''
    Transformer decoder layer.

    Processes sequential inputs with a multi-head self
    attention layer followed by a feed forward layer. 
    Each sub-layer is wrapped with residual connections
    (sub-layer inputs are added to the outputs), optional 
    dropout, and optional layer normalization to 
    stabilize training. 
    '''

    def __init__(self, num_heads, units, drop_rate, normalize=True):
        self.num_heads = num_heads
        self.units = units
        self.drop_rate = drop_rate
        self.normalize = normalize
        self.built = False

    def __call__(self, X, mask=None, inference_mode=False):
        '''
        Args
        ----
        X: np.ndarray 
            Inputs of shape (num_batches, time_steps, d_model).
        mask: np.ndarray
            Attention mask applied during self-attention to prevent 
            the model from attending to future time steps.
        inference_mode: bool
            If True, dropout is disabled.
        '''
        if not self.built:
            self.build()
        # self attention layer with dropout followed by
        # residual connection (adding the input X to the output)
        attn_out = self.attnDropout(self.attn(X, mask), inference_mode) + X
        # apply normalization to maintain consistent
        # and stable output distribution
        if self.normalize:
            attn_out = self.attn_norm(attn_out)
        # feed forward layer with dropout followed by residual
        # connection (adding the input `attn_out` to the output)
        d_out = self.denseDropout(self.dense(attn_out), inference_mode) + attn_out
        # apply normalization to maintain consistent
        # and stable output distribution
        if self.normalize:
            d_out = self.dense_norm(d_out)

        return d_out

    def backward(self, upstream_grad):
        # gradient of dense layer normalization input
        if self.normalize:
            upstream_grad = self.dense_norm.backward(upstream_grad)
        # gradient of dense layer input, feeding in dense dropout backward
        # pass output as input
        ddense = self.dense.backward(self.denseDropout.backward(upstream_grad))
        # add upstream gradients to account for the residual connection
        ddense += upstream_grad
        # gradient of attention normalization input
        if self.normalize:
            ddense = self.attn_norm.backward(ddense)
        # gradient of attention layer input, feeding in attention dropout
        # backward pass output as input
        dattn = self.attn.backward(self.attnDropout.backward(ddense))
        # add upstream gradients to account for the residual connection
        dattn += ddense

        return dattn

    def optimize(self, optimizer, iter_num, optimizer_args={}):
        self.dense.optimize(optimizer, iter_num, **optimizer_args)
        self.attn.optimize(optimizer, iter_num, **optimizer_args)
        if self.normalize:
            self.attn_norm.optimize(optimizer, iter_num, **optimizer_args)
            self.dense_norm.optimize(optimizer, iter_num, **optimizer_args)

    def build(self):
        self.attn = Attention(self.num_heads)
        self.dense = TransformerDense(self.units)
        self.attnDropout = Dropout(self.drop_rate)
        self.denseDropout = Dropout(self.drop_rate)
        if self.normalize:
            self.attn_norm = LayerNorm()
            self.dense_norm = LayerNorm()
        self.built = True
