import numpy as np
from sequential.layers import Layer


class DenseTimeCompress(Layer):
    '''
    Dense layer for compressing the time dimension.

    Takes inputs `X` of shape (batch_size, time_steps, features) and
    projects the time dimension down to `units`, producing outputs of
    shape (batch_size, units, features). This can be used for temporal
    downsampling or feeding data into deeper layers.
    '''

    def __init__(self, units):
        '''
        Args
        ----
        units: int
            Number of time steps in the compressed output.
        '''
        super().__init__()
        assert units > 0
        self.units = units
        self.built = False

    def __call__(self, X):
        '''
        X: np.ndarray
            Inputs of shape (num_batches, time_steps, features).
        '''
        if not self.built:
            self.build(X)
        self.X = X
        # unpack trainable params
        W = self.trainable_params['W']
        # compress inputs along the time dimension from shape
        # (batch_size, time_steps, features) to (batch_size, units, features)
        return np.matmul(W, X)

    def backward(self, upstream_grad):
        # unpack trainable params
        W = self.trainable_params['W']
        # gradients of W in out = np.matmul(W, X),
        # aligning inputs and upstream_grad for matrix multiplication
        # then summing over batches so that dW.shape == W.shape
        dW = np.sum(np.matmul(upstream_grad, self.X.transpose(0, 2, 1)), axis=0)
        # save trainable param gradients for the optimization step
        self.trainable_params_grad = {'W': dW}
        # return input gradients
        return np.matmul(W.T, upstream_grad)

    def build(self, X):
        # initialize weights based on the time dimension of
        # the input
        W = .01 * np.random.randn(self.units, X.shape[1])
        self.trainable_params = {'W': W}
        self.built = True
