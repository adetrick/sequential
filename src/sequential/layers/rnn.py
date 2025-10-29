import numpy as np
from sequential.layers import Layer
from sequential.activations import get_activation
from sequential.initializers import he_normal, glorot_uniform


class RNN(Layer):
    '''
    Recurrent neural net (RNN) layer.

    Processes sequential inputs by maintaining a hidden
    state across time steps. At each step, the hidden state
    is updated based on the current input and the previous
    hidden state. Returns hidden states for all time steps.
    '''

    def __init__(self, units, activation='tanh', use_bias=True, stateful=False, relu_alpha=.01):
        '''
        Args
        ----
        units: int
            Number of input and hidden state weights, a.k.a neurons.
        activation: str
            Activation function to apply ('relu', 'tanh', or None),
            defaults to tanh.
        use_bias: bool
            If True, includes a bias term.
        stateful: bool
            If True, hidden states are preserved across batches/epochs
            instead of being reset each call.
        relu_alpha: float
            If using relu activation, providing an alpha (typically .01
            or .001) can help prevent vanishing gradients.
        '''
        super().__init__()
        assert units > 0
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.stateful = stateful
        self.relu_alpha = relu_alpha
        self.activate = get_activation(activation, relu_alpha)
        self.h = None
        self.built = False

    def __call__(self, X, inference_mode=False):
        '''
        Args
        ----
        X: np.ndarray
            Inputs of shape (num_batches, time_steps, features).
        inference_mode: bool
            If True, the last hidden state isn't saved for
            the next call
        '''
        if not self.built:
            self.build(X)
        # unpack trainable params
        W = self.trainable_params['W']
        b = self.trainable_params['b']
        self.X = X
        self.fcache = []
        self.hidden_states = np.zeros((X.shape[0], X.shape[1], self.units))
        # init the hidden state with cached h if stateful, otherwise zeros
        if self.stateful and self.h is not None:
            # use the n most recent batches, as many as are in the inputs
            h = self.h[-X.shape[0]:]
        else:
            h = np.zeros((X.shape[0], self.units))
        # iterate through the time steps
        for i in range(X.shape[1]):
            # inputs for the current time step
            x = X[:, i, :]
            # concatenate the inputs and hidden state for the current
            # time step to calculate z with a single dot product
            xh = np.concatenate([x, h], axis=1)
            # calc un-activated output for the current time step
            z = np.dot(xh, W)
            if self.use_bias:
                z += b
            # activate output to get current hidden state
            h = self.activate(z)
            # save the hidden state for the current time step
            self.hidden_states[:, i, :] = h
            # cache intermediate variables for backprop
            self.fcache.append({'xh': xh})
        # cache the last hidden state if not in inference mode
        if not inference_mode:
            self.h = h
        # return 3 dimensional hidden states (batch, time_steps, units)
        return self.hidden_states

    def backward(self, upstream_grad):
        '''
        Propagates gradients backwards through the layer,
        iterating over time steps in reverse.

        Since the hidden state influences subsequent
        time steps in the forward pass, its gradient is
        accumulated and passed back at each step.

        Variable names mirror those in the forward pass, with a
        leading 'd' to indicate derivatives. For example:

            z -> dz
            h -> dh
            X -> dX

        This helps track how forward variables contribute to
        the propagated gradients.
        '''
        # unpack trainable params
        W = self.trainable_params['W']
        b = self.trainable_params['b']
        # initialize gradients for the inputs, weights, and bias
        dX = np.zeros_like(self.X)
        dW = np.zeros_like(W)
        db = np.zeros_like(b) if b is not None else None
        # initialize upstream h gradients
        upstream_dh = np.zeros((self.X.shape[0], self.units))
        # iterate backwards through the time steps
        for i in reversed(range(self.X.shape[1])):
            # cached input/hidden state for the current time step
            xh = self.fcache[i]['xh']
            # hidden state for the current time step
            h = self.hidden_states[:, i, :]
            # upstream gradients for the current time step, adding in upstream
            # h gradients
            ug = upstream_grad[:, i, :] + upstream_dh
            # gradient of z in h = self.activate(z), multiplied by upstream gradients
            dz = self.activate.backward(h) * ug
            # gradient of bias in z += bias, adding bias gradients from previous
            # time steps
            if b is not None:
                db += np.sum(dz, axis=0, keepdims=True)
            # gradient of W in z = np.dot(xh, W),
            # adding gradients from previous time steps
            dW += np.dot(xh.T, dz)
            # gradient of xh in z = np.dot(xh, W)
            dxh = np.dot(dz, W.T)
            # xh is a concatenation of x (input) and h (hidden state)
            # at the current time step, so dxh needs to be sliced get
            # the gradients for each
            dx, dh = dxh[:, :self.X.shape[-1]], dxh[:, self.X.shape[-1]:]
            # save input gradients for the current time step
            dX[:, i, :] = dx
            # update upstream hidden gradients for the next time step
            upstream_dh = dh

        # save trainable param gradients for the optimization step
        self.trainable_params_grad = {'W': dW}
        if b is not None:
            self.trainable_params_grad['b'] = db

        # return input gradients
        return dX

    def build(self, X):
        if X.ndim != 3:
            raise ValueError('Input needs to have 3 dimensions: (batch_size, time_steps, features)')
        # initialize weights for the inputs and hidden state based
        # on the last dimension of the input. The number of units
        # is added to the rows to account for the hidden state.
        W = .01 * np.random.randn(X.shape[-1] + self.units, self.units)
        # bias
        b = np.zeros((1, self.units)) if self.use_bias else None
        self.trainable_params = {'W': W, 'b': b}
        self.built = True

    def reset_state(self):
        self.h = None
