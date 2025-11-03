import numpy as np
from sequential.layers import Layer
from sequential.activations import get_activation
from sequential.initializers import glorot_uniform, he_normal


class Dense(Layer):
    '''
    Fully connected dense layer.

    Applies a linear transformation to the inputs `X` (z = X @ W + b) 
    followed by an optional activation function.
    '''

    def __init__(self, units, activation='relu', use_bias=True, relu_alpha=.01):
        '''
        Args
        ----
        units: int
            Number of trainable weights, a.k.a neurons.
        activation: str
            Activation function to apply ('relu', 'tanh', or None).
        use_bias: bool
            If True, includes a bias term.
        relu_alpha: float
            Leak factor for ReLU to help prevent vanishing gradients.
        '''
        assert units > 0
        super().__init__()
        self.units = units
        self.activation = activation
        self.relu_alpha = relu_alpha
        self.activate = get_activation(activation, relu_alpha)
        self.use_bias = use_bias
        self.built = False

    def __call__(self, X):
        '''
        X: np.ndarray
            Inputs of shape (num_batches, time_steps, features).
        '''
        if not self.built:
            self.build(X)
        # unpack trainable params
        W = self.trainable_params['W']
        b = self.trainable_params['b']
        # apply linear projection
        z = np.matmul(X, W)
        # add bias
        if b is not None:
            z += b
        # activation function for nonlinearity
        a = self.activate(z) if self.activate is not None else None
        # cache intermediate variables for backprop
        self.fcache = {'a': a, 'X': X}
        return a if a is not None else z

    def backward(self, upstream_grad):
        '''
        Propagates gradients backwards through the layer.

        Variable names mirror those in the forward pass, with a
        leading 'd' to indicate derivatives. For example:

            W -> dW
            a -> da
            b -> db

        This helps track how forward variables contribute to 
        the propagated gradients.
        '''
        # unpack intermediate variables
        X = self.fcache['X']
        a = self.fcache['a']
        # unpack trainable params
        W = self.trainable_params['W']
        b = self.trainable_params['b']
        if self.activate is not None:
            # calc the gradient of the activation function
            # output
            da = self.activate.backward(a)
            # update the upstream gradients
            upstream_grad *= da
        # gradient of W in X_proj = np.dot(X, W), aligning
        # X and dX for matrix multiplication, then summing over the
        # batches so that dW.shape == W.shape
        dW = np.sum(np.matmul(X.transpose(0, 2, 1), upstream_grad), axis=0)
        # gradient of bias in z += b, which is the batch sum of
        # upstream gradients
        db = None
        if b is not None:
            # sum across the batches and time steps if there are
            # multiple values per time step, otherwise batch sum
            axis = (0, 1) if upstream_grad.shape[-1] > 1 else 0
            db = np.sum(upstream_grad, axis=axis)
            # make sure db.shape == b.shape
            db = db.reshape(b.shape) if db.ndim != b.ndim else db
        # save trainable param gradients for the optimization step
        self.trainable_params_grad = {'W': dW}
        if b is not None:
            self.trainable_params_grad['b'] = db
        # return gradient of the X in z = np.matmul(X, W)
        return np.dot(upstream_grad, W.T)

    def build(self, X):
        # initialize weights based on last dimension of the input
        if self.activation == 'relu':
            W = he_normal(X.shape[-1], self.units) * .01
        else:
            W = glorot_uniform(X.shape[-1], self.units) * .01
        # bias
        b = np.zeros((1, self.units)) if self.use_bias else None
        self.trainable_params = {'W': W, 'b': b}
        self.built = True
