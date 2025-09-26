import numpy as np
from sequential.layers import Layer


class LayerNorm(Layer):
    '''
    Layer normalization layer.

    Normalizes the inputs across a specified axis by subtracting 
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

    def __call__(self, x):
        if not self.built:
            self.build(x)
        # unpack trainable params
        gamma = self.trainable_params['gamma']
        beta = self.trainable_params['beta']
        # calc the mean across the last dimension
        mean = np.mean(x, axis=self.axis, keepdims=True)
        # calc the variance across the last dimension,
        # adding a small epsilon to prevent devision by zero
        var = np.var(x, axis=self.axis, keepdims=True) + self.eps
        # calc the standard deviation across the last dimension
        std = np.sqrt(var)
        # calculate x minus the mean as a separate
        # variable for easier backprop
        x_mu = x - mean
        # normalize
        norm = x_mu / std
        # linearly transform normalized values
        out = (gamma * norm) + beta
        # cache intermediate variables for backprop
        self.fcache = {'norm': norm, 'var': var, 'std': std, 'x': x, 'x_mu': x_mu}

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
        x = self.fcache['x']
        x_mu = self.fcache['x_mu']
        N = x.shape[-1]
        # gradient of gamma in out = (gamma * norm) + beta, summing the gradients
        # over the batch and time step dimensions
        dgamma = np.sum(norm * upstream_grad, axis=(0, 1))
        # gradient of norm in out = (gamma * norm) + beta
        dnorm = gamma * upstream_grad
        # gradient of beta in out = (gamma * norm) + beta, summing the gradients
        # over the batch and time step dimensions
        dbeta = np.sum(upstream_grad, axis=(0, 1))
        # gradient of x_mu in norm = x_mu / std, multiplied by upstream dnorm
        dx_mu = (1 / std) * dnorm
        # gradient of std in norm = x_mu / std, multiplied by upstream dnorm
        # Note: rewrite as (x - mean) * std**-1 before calculating gradient
        dstd = (x_mu) * -1 * std**-2 * dnorm
        # gradient of mean in x_mu = x - mean, multiplied by upstream dx_mu
        dmean = -1 * dx_mu
        # gradient of x in x_mu = x - mean,
        dx = 1 * dx_mu
        # gradient of var in std = np.sqrt(var + 1e-5), multiplied by upstream dstd
        dvar = ((1 / 2) * var**(-1 / 2)) * dstd
        # gradient of x in var = np.var(x, axis=self.axis, keepdims=True), mulitplied by upstream dvar
        # and adding upstream dx
        dx = ((2 / N) * (x_mu) * dvar) + dx
        # gradient of x in mean = np.mean(x, axis=self.axis, keepdims=True), multiplied by upstream dmean
        # and adding upstream dx
        dx = ((1 / N) * dmean) + dx
        # save trainable param gradients for the optimization step
        self.trainable_params_grad = {'gamma': dgamma, 'beta': dbeta}

        return dx

    def build(self, x):
        gamma = np.ones(x.shape[-1])
        beta = np.zeros(x.shape[-1])
        self.trainable_params = {'gamma': gamma, 'beta': beta}
        self.built = True
