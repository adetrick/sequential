import numpy as np
from sequential.layers import Layer


class LayerNorm(Layer):
    '''
    Layer normalization layer.

    Normalizes the inputs `X` across a specified axis by subtracting 
    the mean and dividing by the standard deviation. Then applies 
    learnable linear transformation with scale (gamma) and shift 
    (beta) parameters. This helps stabilize and accelerate training 
    in deep networks.

    Returns normalized and linearly transformed outputs of the same 
    shape as the input.
    '''

    def __init__(self, eps=1e-5, axis=-1):
        '''
        Args
        ----
        eps: float
            Epsilon to prevent division by zero error.
        axis: int
            The axis over which to compute the mean 
            and variance.
        '''
        super().__init__()
        self.eps = eps
        self.axis = axis
        self.built = False

    def __call__(self, X):
        '''
        X: np.ndarray
            Inputs of shape (num_batches, time_steps, features).
        '''
        if not self.built:
            self.build(X)
        # unpack trainable params
        gamma = self.trainable_params['gamma']
        beta = self.trainable_params['beta']
        # calc the mean across the last dimension
        mean = np.mean(X, axis=self.axis, keepdims=True)
        # calc the variance across the last dimension,
        # adding a small epsilon to prevent devision by zero
        var = np.var(X, axis=self.axis, keepdims=True) + self.eps
        # calc the standard deviation across the last dimension
        std = np.sqrt(var)
        # calculate x minus the mean as a separate
        # variable for easier backprop
        X_mu = X - mean
        # normalize
        norm = X_mu / std
        # linearly transform normalized values
        out = (gamma * norm) + beta
        # cache intermediate variables for backprop
        self.fcache = {'norm': norm, 'var': var, 'std': std, 'X': X, 'X_mu': X_mu}

        return out

    def backward(self, upstream_grad):
        '''
        Propagates gradients backwards through the layer.

        Variable names mirror those in the forward pass, with a
        leading 'd' to indicate derivatives. For example:

            norm -> dnorm
            var -> dvar
            x -> dx

        This helps track how forward variables contribute to 
        the propagated gradients.
        '''
        # unpack trainable params
        gamma = self.trainable_params['gamma']
        # unpack intermediate variables
        norm = self.fcache['norm']
        var = self.fcache['var']
        std = self.fcache['std']
        X = self.fcache['X']
        X_mu = self.fcache['X_mu']
        N = X.shape[-1]
        # gradient of gamma in out = (gamma * norm) + beta, summing the gradients
        # over the batch and time step dimensions
        dgamma = np.sum(norm * upstream_grad, axis=(0, 1))
        # gradient of norm in out = (gamma * norm) + beta
        dnorm = gamma * upstream_grad
        # gradient of beta in out = (gamma * norm) + beta, summing the gradients
        # over the batch and time step dimensions
        dbeta = np.sum(upstream_grad, axis=(0, 1))
        # gradient of X_mu in norm = X_mu / std, multiplied by upstream dnorm
        dX_mu = (1 / std) * dnorm
        # gradient of std in norm = X_mu / std, multiplied by upstream dnorm
        # Note: rewrite as (X - mean) * std**-1 before calculating gradient
        dstd = (X_mu) * -1 * std**-2 * dnorm
        # gradient of mean in X_mu = X - mean, multiplied by upstream dX_mu
        dmean = -1 * dX_mu
        # gradient of X in X_mu = X - mean,
        dX = 1 * dX_mu
        # gradient of var in std = np.sqrt(var + 1e-5), multiplied by upstream dstd
        dvar = ((1 / 2) * var**(-1 / 2)) * dstd
        # gradient of X in var = np.var(X, axis=self.axis, keepdims=True),
        # multiplied by upstream dvar and adding upstream dx
        dX = ((2 / N) * (X_mu) * dvar) + dX
        # gradient of X in mean = np.mean(X, axis=self.axis, keepdims=True),
        # multiplied by upstream dmean and adding upstream dX
        dX = ((1 / N) * dmean) + dX
        # save trainable param gradients for the optimization step
        self.trainable_params_grad = {'gamma': dgamma, 'beta': dbeta}

        return dX

    def build(self, x):
        gamma = np.ones(x.shape[-1])
        beta = np.zeros(x.shape[-1])
        self.trainable_params = {'gamma': gamma, 'beta': beta}
        self.built = True
