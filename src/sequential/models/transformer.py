import numpy as np
from sequential.models import Model
from sequential.models import Decoder
from sequential.layers import Dense


class Transformer(Model):
    '''
    Decoder-only Transformer model for sequential inputs.

    Wraps a Decoder object and provides:

        - forward prediction with self attention and feed forward layers
        - backpropagation through the decoder, output, embedding, and 
          time compression layers
        - parameter optimization with configurable optimizers

    Extends Model and implements the core methods required
    for training and inference.
    '''

    def __init__(
            self,
            d_model=50,
            num_heads=2,
            num_decoder_layers=3,
            units=64,
            normalize=False,
            loss='mse',
            drop_rate=0,
            optimizer='adam',
            optimizer_args=None,
            apply_positional_encoding=False):
        '''
        Args
        ----
        d_model: int 
            Dimensionality of the model's vector representations (i.e. embeddings)
            of the raw inputs. Raw inputs are projected into this dimension 
            before being fed into the Decoder.
        num_heads: int
            Number of heads in each self-attention block.
        num_decoder_layers: int
            Number of stacked decoder layers in the Transformer.
        units: int
            Number of units in the feed-forward sublayer within each
            decoder layer.
        normalize: bool
            If True, applies layer normalization after each decoder sublayer.
        loss: str
            Loss function to minimize. Options: 'mse', 'mae'.
        drop_rate: float
            Dropout rate for regularization. Must be in the range 
            0 <= drop_rate < 1. Dropout is only applied during training.
        optimizer: str
            Optimization algorithm for updating parameters. Options: 
            'gd' (gradient descent), 'adam'.
        optimizer_args: dict
            Dictionary of hyperparameters passed to the optimizer (e.g., {'alpha': 0.001}).
        apply_positional_encoding: bool
            If True, positional encodings will be added to the embedded inputs.
        '''
        self.d_model = d_model
        self.num_heads = num_heads
        self.decoder = Decoder(max(num_decoder_layers, 1), num_heads, units, drop_rate, normalize)
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args if optimizer_args is not None else {}
        self.apply_positional_encoding = apply_positional_encoding
        self.embedding_matrix = None
        # trainable embedding layer to project the features dimension of the
        # inputs from (batch_size, time_steps, features) --> (batch_size, time_steps, d_model)
        self.embed_layer = Dense(self.d_model, activation=None,
                                 use_bias=True) if self.d_model > 1 else None
        # output layer to project each embedded d_model dimension down to
        # a single output value
        self.output_layer = Dense(1, activation=None, use_bias=False)
        self.mask = None
        self.validate()

    def predict(self, X, inference_mode):
        # create a self attention mask for the decoder
        if self.mask is None:
            self.mask = self.generate_mask(X.shape[1], self.num_heads > 1)
        # feed inputs into the decoder
        dec_out = self.decoder(X, self.mask, inference_mode)
        # project d_model dimension down to a single output dimension
        out = self.output_layer(dec_out)
        # return output through a dense time compressor if the
        # target has fewer time steps than the inputs, otherwise
        # return the output directly
        return self.dtc(out) if self.dtc is not None else out

    def back_propagation(self, pred, dec_target):
        # backprop through the cost function
        dcost = self.dcost(dec_target, pred)
        # backprop through the dense time compressor
        if self.dtc is not None:
            dcost = self.dtc.backward(dcost)
        # backprop through the output layer
        doutput_layer = self.output_layer.backward(dcost)
        # backprop through the decoder
        ddecoder = self.decoder.backward(doutput_layer)
        # backprop through the embedding layer
        if self.embed_layer is not None:
            dembed_layer = self.embed_layer.backward(ddecoder)

    def generate_mask(self, seq_len, multihead=False):
        '''
        Creates a self-attention mask
        '''
        shape = (1, 1, seq_len, seq_len) if multihead else (1, seq_len, seq_len)
        mask = np.tril(np.ones(shape))
        # 0.0 for allowed, -inf for blocked
        mask = np.where(mask == 1, 0.0, -np.inf)
        return mask

    def optimize(self, iter_num):
        '''
        Runs the optimize step for updating parameters in the decoder,
        output, embedding, and time compression layers
        '''
        self.output_layer.optimize(self.optimizer, iter_num, **self.optimizer_args)
        self.decoder.optimize(self.optimizer, iter_num, self.optimizer_args)
        if self.embed_layer is not None:
            self.embed_layer.optimize(self.optimizer, iter_num, **self.optimizer_args)
        if self.dtc is not None:
            self.dtc.optimize(self.optimizer, iter_num, **self.optimizer_args)

    def validate(self):
        '''
        Validates init params
        '''
        if self.num_heads > 0 and self.d_model % self.num_heads != 0:
            raise ValueError(
                "D model needs to be divided into equal parts for multi-head attention, choose a different number of heads")
