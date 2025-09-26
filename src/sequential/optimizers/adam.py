import numpy as np


class AdamOptimizer:
    '''     
    Implements the Adam (Adaptive Moment Estimation) optimization algorithm.

    Adam adaptively adjusts the learning rate for each parameter based 
    on estimates of the first and second moments of past gradients:

        - First moment (m): exponentially decaying average of past gradients.
        - Second moment (v): exponentially decaying average of past squared gradients.

    Bias correction is applied to both moment estimates to counteract 
    their initialization at zero.
    '''

    def __init__(self, b1=.9, b2=.999, alpha=.001, eps=1e-8):
        '''
        Args
        ----
        b1: float
            Decay rate for the moving average of the gradients (momentum).
        b2: float
            Decay rate for the moving average of the squared gradients 
            (second momentum).
        alpha: float
            Learning rate for parameter updates.
        eps: float
            Small constant to prevent division by zero.
        '''
        self.b1 = b1
        self.b2 = b2
        self.alpha = alpha
        # a small epsilon value is used to prevent division by zero
        # when returning the updated parameters in optimize()
        self.eps = eps
        # first momentum estimate `m` is the moving average of
        # past gradients. It helps smooth the optimization path for
        # faster convergence
        self.m = None
        # second momentum estimate `v` is the moving average of
        # past squared gradients. It helps overcome diminishing
        # learning rates
        self.v = None
        self.built = False

    def optimize(self, params, gradients, iter_=None):
        '''
        Updates parameters using the Adam algorithm.

        Args
        ----
        params: array
            Trainable model parameters.
        gradients: array
            Gradients of the parameters with respect to the loss, 
            same shape as `params`.
        iter_: int
            Current iteration number, used for bias correction of 
            moment estimates.
        '''
        if not self.built:
            self.build(gradients)
        # update the first and second momentum estimates
        self.m = (self.b1 * self.m) + ((1 - self.b1) * gradients)
        self.v = (self.b2 * self.v) + ((1 - self.b2) * gradients**2)
        m_hat = self.m
        v_hat = self.v
        # compute bias corrected m and v estimates. They're biased
        # towards zero (especially during the first iterations) due
        # to being initialized at zero.
        if iter_ is not None:
            m_hat = self.m / (1 - self.b1**(iter_ + 1))
            v_hat = self.v / (1 - self.b2**(iter_ + 1))
        # return the updated parameters
        return params - (self.alpha * (m_hat / (np.sqrt(v_hat + self.eps))))

    def build(self, gradients):
        self.m = np.zeros_like(gradients)
        self.v = np.zeros_like(gradients)
        self.built = True
