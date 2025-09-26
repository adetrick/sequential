class GradientDescent:
    '''
    Gradient descent with optional momentum.

    Args
    ----
    alpha: float
        Learning rate for parameter optimization.
    m: float
        Momentum term between 0 and 1, a lower factor discounts
        older observations faster.
    '''

    def __init__(self, alpha=.001, m=None):
        self.alpha = alpha
        self.m = m  # momentum
        self.v = 0  # velocity

    def optimize(self, params, gradients):
        grads = gradients
        # if there's momentum, update the velocity term
        if self.m:
            self.v = (self.m * self.v) + ((1 - self.m) * gradients)
            # set the gradients to the velocity
            grads = self.v

        return params - (grads * self.alph)
