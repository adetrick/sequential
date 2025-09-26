import numpy as np
from sequential.layers import Layer
from sequential.activations import get_activation


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

    def __call__(self, inputs, inference_mode=False):
        if not self.built:
            self.build(inputs)
        # unpack trainable params
        weights = self.trainable_params['W']
        bias = self.trainable_params['b']
        self.inputs = inputs
        self.fcache = []
        self.hidden_states = np.zeros((inputs.shape[0], inputs.shape[1], self.units))
        # init the hidden state with cached h if stateful, otherwise zeros
        if self.stateful and self.h is not None:
            # use the n most recent batches, as many as are in the inputs
            h = self.h[-inputs.shape[0]:]
        else:
            h = np.zeros((inputs.shape[0], self.units))
        # iterate through the time steps
        for i in range(inputs.shape[1]):
            # inputs for the current time step
            x = inputs[:, i, :]
            X = np.concatenate([h, x], axis=1)
            # calc un-activated output for the current time step
            z = np.dot(X, weights)
            if self.use_bias:
                z += bias
            # activate output to get current hidden state
            h = self.activate(z)
            # save the hidden state for the current time step
            self.hidden_states[:, i, :] = h
            # cache intermediate variables for backprop
            self.fcache.append({'X': X})
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
        weights = self.trainable_params['W']
        bias = self.trainable_params['b']
        # initialize gradients for the inputs, weights, and bias
        grad_inputs = np.zeros_like(self.inputs)
        grad_weights = np.zeros_like(weights)
        grad_bias = np.zeros_like(bias) if bias is not None else None
        # initialize upstream h gradients
        upstream_dh = np.zeros((self.inputs.shape[0], self.units))
        # iterate backwards through the time steps
        for i in reversed(range(self.inputs.shape[1])):
            # cached input for the current time step
            X = self.fcache[i]['X']
            # hidden state for the current time step
            h = self.hidden_states[:, i, :]
            # upstream gradients for the current time step, adding in upstream
            # h gradients
            ug = upstream_grad[:, i, :] + upstream_dh
            # gradients of z in h = self.activate(z), multiplied by upstream gradients
            dz = self.activate.backward(h) * ug
            # gradient of bias in z += bias, adding bias gradients from previous
            # time steps
            if bias is not None:
                grad_bias += np.sum(dz, axis=0, keepdims=True)
            # gradients of weights in z = np.dot(X, weights),
            # adding gradients from previous time steps
            grad_weights += np.dot(X.T, dz)
            # gradients of X in z = np.dot(X, weights)
            dX = np.dot(dz, weights.T)
            # X is a concatenation of h (hidden state) and x
            # (input), so dX needs to be sliced get the gradients
            # for each
            dh, dx = dX[:, :self.units], dX[:, self.units:]
            # save input gradients for the current time step
            grad_inputs[:, i, :] = dx
            # update upstream hidden gradients for the next time step
            upstream_dh = dh

        # save trainable param gradients for the optimization step
        self.trainable_params_grad = {'W': grad_weights}
        if bias is not None:
            self.trainable_params_grad['b'] = grad_bias

        # return input gradients
        return grad_inputs

    def build(self, inputs):
        if inputs.ndim != 3:
            raise ValueError('Input needs to have 3 dimensions: (batch_size, time_steps, features)')
        # initialize weights for the inputs and hidden state based
        # on the last dimension of the input. The number of units
        # is added to the rows to account for the hidden state.
        weights = .01 * np.random.randn(inputs.shape[-1] + self.units, self.units)
        bias = np.zeros((1, self.units)) if self.use_bias else None
        self.trainable_params = {'W': weights, 'b': bias}
        self.built = True

    def reset_state(self):
        self.h = None
