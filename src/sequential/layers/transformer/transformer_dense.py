import numpy as np
from sequential.layers import Layer
from sequential.activations import Relu


class TransformerDense(Layer):
    '''
    A fully connected feed forward layer consisting of
    two linear transformations with a ReLU activation 
    in between (from the 'Attention is All You Need' paper)
    '''

    def __init__(self, units):
        assert units > 0
        super().__init__()
        self.units = units
        self.relu = Relu()
        self.built = False

    def __call__(self, X):
        '''
        X: np.ndarray 
            Inputs of shape (num_batches, time_steps, d_model).
        '''
        if not self.built:
            self.build(X)
        # unpack trainable params
        W1 = self.trainable_params['W1']
        b1 = self.trainable_params['b1']
        W2 = self.trainable_params['W2']
        b2 = self.trainable_params['b2']
        # apply first linear projection
        z = np.dot(X, W1) + b1
        # apply ReLU nonlinearity
        a = self.relu(z)
        # project back to original model dimension
        a2 = np.dot(a, W2) + b2
        # cache intermediate variables for backprop
        self.fcache = {'a': a, 'X': X}

        return a2

    def backward(self, upstream_grad):
        '''
        Propagates gradients backwards through the layer.

        Variable names mirror those in the forward pass, with a
        leading 'd' to indicate derivatives. For example:

            a -> da
            W1 -> dW1
            X -> dX

        This helps track how forward variables contribute to 
        the propagated gradients.
        '''
        # unpack trainable params
        W1 = self.trainable_params['W1']
        W2 = self.trainable_params['W2']
        # unpack intermediate variables
        a = self.fcache['a']
        X = self.fcache['X']
        # gradient of a in np.dot(a, W2) + b2
        da = np.dot(upstream_grad, W2.T)
        # gradient of W2 in np.dot(a, W2) + b2, align a and
        # upstream_grad for matrix multiplication then sum
        # over the batches so that dW2.shape == W2.shape
        dW2 = np.sum(np.matmul(a.transpose(0, 2, 1), upstream_grad), axis=0)
        # gradient of b2 in np.dot(a, W2) + b2, which is
        # the batch and time step sum of upstream gradients
        db2 = np.sum(upstream_grad, axis=(0, 1))[np.newaxis, :]
        # gradient of ReLU nonlinearity, multiplied
        # by upstream da
        drelu_ = self.relu.backward(a) * da
        # gradient of W1 in np.dot(X, W1) + b1, align
        # X and drelu_ for matrix multiplication, then
        # sum over batches so that dW1.shape == W1.shape
        dW1 = np.sum(np.matmul(X.transpose(0, 2, 1), drelu_), axis=0)
        # gradient of b1 in np.dot(X, W1) + b1, which is
        # the batch and time step sum of upstream gradients
        db1 = np.sum(drelu_, axis=(0, 1))[np.newaxis, :]
        # gradient of X in np.dot(X, W1) + b1
        dX = np.dot(drelu_, W1.T)
        # save trainable param gradients for the optimization step
        self.trainable_params_grad = {'W1': dW1, 'W2': dW2, 'b2': db2, 'b1': db1}
        # return input gradients
        return dX

    def build(self, x):
        # initialize weights based on last dimension of the input (d_model)
        d_model = x.shape[-1]
        # weights for first linear transform
        W1 = np.random.randn(d_model, self.units) * .01
        # bias for first linear transform
        b1 = np.zeros((1, self.units))
        # weights for second linear transform
        W2 = np.random.randn(self.units, d_model) * .01
        # bias for second linear transform
        b2 = np.zeros((1, d_model))
        self.trainable_params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        self.built = True
