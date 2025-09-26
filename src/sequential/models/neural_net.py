from sequential.models import Model
from sequential.layers import Dense, RNN, LSTM, Dropout, LayerNorm


class NeuralNet(Model):
    '''
    Neural network container for sequential layer objects.

    Wraps layers such as Dense, RNN, LSTM, Dropout, and LayerNorm, 
    and provides:

        - forward prediction across all layers
        - backpropagation through the network
        - parameter optimization with configurable optimizers

    Extends Model and implements the core methods required
    for training and inference.
    '''

    def __init__(
            self,
            layers,
            loss='mse',
            optimizer='adam',
            optimizer_args=None):
        '''
        Args
        ----
        layers: list
            List of initialized layer objects. Options: Dense,
            RNN, LSTM, Dropout, LayerNorm. Must start with a 
            Dense, RNN, or LSTM layer.
        loss: str
            Loss function to minimize. Options: 'mse', 'mae'.
        optimizer: str 
            Optimization algorithm for updating parameters. Options: 
            'gd' (gradient descent), 'adam'.
        optimizer_args: dict
            Dictionary of hyperparameters passed to the optimizer (e.g., {'alpha': 0.001}).
        '''
        super().__init__()
        self.layers = self.process_layers(layers)
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args if optimizer_args is not None else {}

    def predict(self, inputs, inference_mode):
        for layer in self.layers:
            if isinstance(layer, (RNN, LSTM, Dropout)):
                out = layer(inputs, inference_mode=inference_mode)
            else:
                out = layer(inputs)
            # feed output from this layer into the next one
            inputs = out
        # return output through the dense time compressor if the
        # target has fewer time steps than the inputs, otherwise
        # return the output directly
        return self.dtc(out) if self.dtc is not None else out

    def optimize(self, iter_num):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                continue
            layer.optimize(self.optimizer, iter_num, **self.optimizer_args)
        if self.dtc is not None:
            self.dtc.optimize(self.optimizer, iter_num, **self.optimizer_args)

    def back_propagation(self, pred, y):
        # initialize upstream gradients with the partial derivative
        # of the cost function with respect to the prediction
        upstream_grad = self.dcost(y, pred)
        # backprop through the dense time compressor
        if self.dtc is not None:
            upstream_grad = self.dtc.backward(upstream_grad)
        # backprop through the layers in reverse order
        for layer in reversed(self.layers):
            upstream_grad = layer.backward(upstream_grad)

    def process_layers(self, layers):
        if len(layers) == 0:
            raise ValueError("At least one layer must be provided")

        if isinstance(layers[0], (Dropout, LayerNorm)):
            raise ValueError("First layer must be a Dense, RNN, or LSTM layer")

        for i, layer in enumerate(layers):
            if type(layer) not in [Dense, LSTM, RNN, Dropout, LayerNorm]:
                raise ValueError(
                    f"Layer {i} must be an instantiated Dense, LSTM, RNN, Dropout, or LayerNorm layer")

        # add an output layer if one isn't provided
        if layers[-1].units > 1:
            layers.append(Dense(1, activation=None, use_bias=False))

        return layers
