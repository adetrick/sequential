import numpy as np


class Softmax:
    '''
    Applies the Softmax activation function.

    Transform input vectors into probability distributions. 
    Each output value is between (0, 1) and the values sum to 
    1 along the last axis.
    '''

    def __call__(self, x):
        # subtract the max to prevent overflow when
        # calculating exp(x)
        x -= np.max(x, axis=-1, keepdims=True)
        # calc the exponent for each element
        exp = np.exp(x)
        # return probabilities: exponent of x divided
        # by the sum of exponents
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def backward(self, soft_out, upstream_grad):
        '''
        Derivative of Softmax output

        Args
        ----
        soft_out: np.ndarray
            Softmax output.
        upstream_grad: np.ndarray
            Upstream gradients from backpropagation.

        Returns
        -------
        gradients w.r.t the Softmax input

        Notes
        -----
        The academically proper way to compute the gradient of Softmax 
        is by calculating its Jacobian:

            J = diag(s) - s sᵀ

        where s is the Softmax output vector. For an input gradient vector `g`, 
        the gradient w.r.t. input is:

            grad_out = J @ g

        This requires constructing an (N,N) Jacobian for each Softmax vector, 
        which is computationally expensive. See `dsoftmax_jac()` for 
        a full implementation.

        However, the same result can be derived more efficiently:

            grad_out = (s * g) - s * sum(s * g)

        or equivalently (after factoring out s):

            grad_out = s * (g - sum(s * g))

        This formula, which is implemented by modern frameworks like 
        Tensorflow [2], avoids explicitly building the Jacobian for 
        increased efficiency.

        Quick intuition:

        - Start with grad_out = J @ g
        - Substitute the Jacobian: grad_out = (diag(s) - s sᵀ) @ g
        - Distribute g: grad_out = (diag(s) @ g) - (s sᵀ @ g)
        - Simplify each term:
            * diag(s) @ g  -->  s * g  [3]
            * (s sᵀ) @ g  -->  s * sum(s * g) [4]
        - Combine: grad_out = (s * g) - (s * sum(s * g))
        - Factor out s: grad_out = s * (g - sum(s * g))

        References
        ----------
        (1) https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        (2) https://github.com/tensorflow/tensorflow/blob/5bc9d26649cca274750ad3625bd93422617eed4b/tensorflow/python/ops/nn_grad.py#L282
        (3) np.diagflat(s) outputs a diagonal matrix of zeros with s on the diagonal. Multiplying by g yields s * g.
        (4) https://youtu.be/27vT-NWuw0M?si=WB-GrPPFtJav0o1P&t=479
        '''

        # sum up the multiplication of the Softmax out values and the
        # upstream gradient along the last axis (the axis containing the
        # Softmax output vectors)
        sum_soft_grad = np.sum(soft_out * upstream_grad, axis=-1, keepdims=True)

        return soft_out * (upstream_grad - sum_soft_grad)


def dsoftmax_jac(soft_out, upstream_grad):
    '''
    Example of how the softmax output gradients are calculated 
    with the slower Jacobian-based method, see Softmax.backward 
    'for the efficient equivalent.
    '''
    # unpack the first three dimensions to iterate through, note that
    # the fourth dimension contains the softmax output vectors
    batches, heads, seq_len = soft_out.shape[0], soft_out.shape[1], soft_out.shape[2]
    # initialize the gradient outputs
    grad_out = np.zeros_like(soft_out)
    # iterate through the first three dimensions
    for b in range(batches):
        for h in range(heads):
            for s in range(seq_len):
                # make sure the softmax vector is shaped (N, 1)
                # so that the dot product below produces a matrix
                # of (N, N) in order for the subtraction to work
                # with the Jacobian
                soft_vec = soft_out[b, h, s].reshape(-1, 1)
                # calculate the Jacobian
                jac = np.diagflat(soft_vec) - np.dot(soft_vec, soft_vec.T)
                # multiply the Jacobian by the corresponding vector of
                # upstream gradients, which results in a 1D vector
                # of softmax output gradients
                grad_out[b, h, s] = np.dot(jac, upstream_grad[b, h, s])

    return grad_out


class Tanh:
    '''
    Applies the Tanh (hyperbolic tangent) activation function.

    Transforms inputs to the range (-1, 1), centered around 0.
    The use of both positive and negative outputs stabilizes 
    training and can represent state changes more effectively 
    in recurrent neural networks. 
    '''

    def __call__(self, x):
        return np.tanh(x)

    def backward(self, tanh_out):
        return 1 - tanh_out**2


class Sigmoid:
    '''
    Applies the sigmoid activation function.

    Transforms input to the range (0, 1). The output is .5
    when the input is 0 and approaches 1 or 0 as the input 
    increases or decreases, respectively. Because of this 
    bounded range, sigmoid is often used to model probabilities.
    '''

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, sig_out):
        return sig_out * (1 - sig_out)


class Relu:
    '''
    Applies the rectified linear unit (ReLU) activation function.

    ReLU outputs the input directly if it is positive, and 0 
    otherwise. This supports faster convergence and reduces the 
    vanishing gradient problem seen in sigmoid/tanh. However, 
    units stuck at 0 may not recover, which is known as the 
    "dying ReLU" problem. This can be mitigated with an `alpha` term.
    '''

    def __init__(self, alpha=None):
        '''
        Args
        ----
        alpha: float
            Applies the leaky ReLU technique to mitigate the dying ReLU 
            problem with the standard ReLU formula since setting x[x < 0] = 0 
            results in gradients of 0 that prevent downstream weights and biases 
            from being updated. Standard alpha values are .01 or .001.
        '''
        self.alpha = alpha

    def __call__(self, x):
        # apply the leaky relu variant if there's
        # an alpha value
        if self.alpha is not None:
            return np.maximum((x * self.alpha), x)

        return np.maximum(0, x)

    def backward(self, x):
        # the gradient of the ReLU activation output is 1
        # for values greater than 0
        dx = np.ones_like(x)
        dx[x < 0] = self.alpha if self.alpha is not None else 0
        return dx


def get_activation(name, relu_alpha=None):
    if name == 'relu':
        return Relu(alpha=relu_alpha)
    elif name == 'softmax':
        return Softmax()
    elif name == 'tanh':
        return Tanh()
