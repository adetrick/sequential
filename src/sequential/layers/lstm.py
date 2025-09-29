import numpy as np
from sequential.layers import Layer
from sequential.activations import Sigmoid, Tanh


class LSTM(Layer):
    '''
    Long short-term memory (LSTM) layer.

    Processes sequential inputs using input, forget, 
    and output gates to control information flow while
    maintaining hidden and cell states across time steps. 
    At each step, the layer updates the hidden and cell states 
    based on the current input and previous states. Returns 
    hidden states for all time steps.
    '''

    def __init__(self, units, use_bias=True, stateful=False):
        '''
        Args
        ----
        units: int
            Number of input and hidden state weights, a.k.a neurons.
        use_bias: bool
            If True, includes a bias term.
        stateful: bool
            If True, hidden and cell states are preserved across 
            batches/epochs instead of being reset each call.
        '''
        super().__init__()
        assert units > 0
        self.units = units
        self.use_bias = use_bias
        self.stateful = stateful
        self.h = None
        self.c = None
        self.bias = None
        self.built = False
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()

    def __call__(self, X, inference_mode=False):
        '''
        Args
        ----
        X: np.ndarray 
            Inputs of shape (num_batches, time_steps, features).
        inference_mode: bool
            If True, the last hidden and cell states aren't saved 
            for the next call
        '''
        if not self.built:
            self.build(X)
        self.X_shape = X.shape
        self.fcache = []
        # unpack trainable params
        W = self.trainable_params['W']
        bias = self.trainable_params['b']
        # init hidden state
        hidden_states = np.zeros((X.shape[0], X.shape[1], self.units))
        h = None
        c = None
        if self.stateful and self.h is not None and self.c is not None:
            # use the n most recent batches, as many as are in the inputs
            h, c = self.h[-X.shape[0]:], self.c[-X.shape[0]:]
        else:
            h = np.zeros((X.shape[0], self.units))
            c = np.zeros((X.shape[0], self.units))
        # iterate through the time steps
        for i in range(X.shape[1]):
            # inputs for the current time step
            x = X[:, i, :]
            # concatenate the hidden state and the
            # inputs from the current time step to
            # calculate z with a single dot product
            xh = np.concatenate([x, h], axis=1)
            z = np.dot(xh, W)
            if bias is not None:
                z += bias
            # forget gate
            fg = self.sigmoid(z[:, :self.units])
            # input gate
            ig = self.sigmoid(z[:, self.units:self.units * 2])
            # candidate values
            cand = self.tanh(z[:, self.units * 2:self.units * 3])
            # output gate
            og = self.sigmoid(z[:, self.units * 3:self.units * 4])
            # update the cell state
            c = (ig * cand) + (fg * c)
            # cell state state activation
            ca = self.tanh(c)
            # update the hidden state for current time step
            h = og * ca
            # save hidden state
            hidden_states[:, i, :] = h
            # cache intermediate variables at current time step for backprop
            self.fcache.append({'fg': fg, 'ig': ig, 'cand': cand,
                               'og': og, 'xh': xh, 'ca': ca, 'c': c})
        # cache the last hidden/cell states for the next call if not
        # in inference mode
        if not inference_mode:
            self.h, self.c = h, c
        # return 3 dimensional hidden states (batch_size, time_steps, units)
        return hidden_states

    def backward(self, upstream_grad):
        '''
        Propagates gradients backwards through the layer,
        iterating over time steps in reverse.

        Since hidden and cell states influence subsequent
        time steps in the forward pass, their gradients are 
        accumulated and passed back at each step.

        Variable names mirror those in the forward pass, with a
        leading 'd' to indicate derivatives. For example:

            fg -> dfg
            h -> dh
            inputs -> dinputs

        This helps track how forward variables contribute to 
        the propagated gradients.
        '''
        # unpack trainable params
        W = self.trainable_params['W']
        b = self.trainable_params['b']
        # initialize gradients for the weights with zeros,
        # gradients will be cumulatively summed while iterating
        # backwards through the time steps
        dW = np.zeros_like(W)
        dX = np.zeros(self.X_shape)
        if b is not None:
            db = np.zeros_like(b)
        # initialize the upstream cell state and hidden gradients
        upstream_dc = np.zeros((self.X_shape[0], self.units))
        upstream_dh = np.zeros((self.X_shape[0], self.units))
        # iterate backwards through the time steps
        for i in reversed(range(self.X_shape[1])):
            # unpack intermediate variables for current time step
            cache = self.fcache[i]
            xh = cache['xh']  # concatenated input and hidden state
            fg = cache['fg']  # forget gate
            ig = cache['ig']  # input gate
            cand = cache['cand']  # candidate values
            og = cache['og']  # output gate
            ca = cache['ca']  # cell state activation
            c = cache['c']  # cell state
            # cell state from previous time step
            c_prev = self.fcache[i - 1]['c'] if i > 0 else np.zeros_like(c)
            # upstream gradients at the current time step, adding in
            # upstream dh
            ug = upstream_grad[:, i, :] + upstream_dh
            # gradient of og in h = og * ca, multiplying by upstream gradients
            dog = ca * ug
            # gradient of ca in h = og * ca, multiplying by upstream gradients
            # and adding upstream cell state gradients
            dca = og * ug + upstream_dc
            # gradient of c state activation, multiplied by
            # upstream dca
            dca = self.tanh.backward(ca) * dca
            # gradient of ig in c = (ig * cand) + (fg * c)
            dig = cand * dca
            # gradient of cand in c = (ig * cand) + (fg * c)
            dcand = ig * dca
            # gradient of fg in c = (ig * cand) + (fg * c)
            dfg = c_prev * dca
            # gradient of c in c = (ig * cand) + (fg * c), which becomes
            # the upstream cell state gradient for the next time step
            upstream_dc = fg * dca
            # gradient of input gate activation, multiplied by upstream dig
            dig *= self.sigmoid.backward(ig)
            # gradient of forget gate activation, multiplied by upstream dfg
            dfg *= self.sigmoid.backward(fg)
            # gradient of candidate value activation, multiplied by upstream dcand
            dcand *= self.tanh.backward(cand)
            # gradient of output gate activation, multiplied by upstream dog
            dog *= self.sigmoid.backward(og)
            # stack activation function gradients, maintaining the same
            # gate order as the forward pass
            dz = np.hstack([dfg, dig, dcand, dog])
            # gradient of W in z = np.dot(xh, W)
            dW += np.dot(xh.T, dz)
            # gradient of b in z = np.dot(xh, W) + b
            if b is not None:
                db += np.sum(dz, axis=0, keepdims=True)
            # gradient of xh in z = np.dot(xh, W)
            dxh = np.dot(dz, W.T)
            # xh is a concatenation of x (input) and h (hidden state)
            # at the current time step, so dxh needs to be sliced get
            # the gradients for each
            dx, dh = dxh[:, :self.units], dxh[:, self.units:]
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
        # initialize weights for the input, hidden state, and gates
        # based on the last dimension of the input. The number of
        # units is added to the rows to account for the hidden state,
        # and the columns are multiplied by 4 to include the forget, input,
        # cell, and output gates.
        W = .01 * np.random.randn(X.shape[-1] + self.units, self.units * 4)
        # bias
        b = None
        if self.use_bias:
            b = np.zeros((1, self.units * 4))
            # init the forget bias with ones to encourage the model
            # to retain information early on
            b[:self.units] = 1
        self.trainable_params = {'W': W, 'b': b}
        self.built = True

    def reset_state(self):
        self.c = None
        self.h = None
