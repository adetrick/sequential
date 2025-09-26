import numpy as np


class Dropout:
    '''
    Dropout layer for regularization.

    Randomly sets a fraction of input units to zero with probability 
    equal to `drop_rate`. The remaining units are scaled by 
    1 / (1 - drop_rate) to preserve the sum of inputs. At inference, 
    dropout is disabled and inputs are returned unchanged.
    '''

    def __init__(self, drop_rate=.1):
        '''
        Args
        ----
        drop_rate: float 
            Probability of dropping a unit/neuron, must be in the range
            of (0, 1). A value of 0 disables dropout.
        '''
        assert 0 <= drop_rate < 1
        self.drop_rate = drop_rate

    def __call__(self, X, inference_mode=False):
        '''
        Args
        ----
        X: np.ndarray
            Input data to apply the dropout to.
        inference_mode: bool
            If True, dropout is disabled and inputs are returned 
            unchanged.
        '''
        # no dropout if the rate is 0 or for inference mode
        if self.drop_rate == 0 or inference_mode:
            return X
        # create a boolean mask where True keeps and False drops
        mask = np.random.rand(*X.shape) > self.drop_rate
        # cache for backprop
        self.fcache = {'mask': mask}
        # apply the mask to the inputs and scale the rest (to preserve
        # the sum of the inputs)
        return X * mask / (1 - self.drop_rate)

    def backward(self, upstream_grad):
        # simply return the upstream gradients if there isn't a
        # forward cache (i.e. the drop rate is 0)
        if not hasattr(self, 'fcache'):
            return upstream_grad
        return upstream_grad * self.fcache['mask'] / (1 - self.drop_rate)
