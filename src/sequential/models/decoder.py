from sequential.layers.transformer import DecoderLayer


class Decoder:
    '''
    Transformer decoder

    Stacks multiple decoder layers and manages their execution
    during training and inference:

        - forward pass through all decoder layers
        - backward propagation of gradients
        - optimization of trainable parameters across layers
    '''

    def __init__(self, num_layers, num_heads, units, drop_rate, normalize):
        self.layers = [DecoderLayer(num_heads, units, drop_rate, normalize=normalize)
                       for i in range(num_layers)]

    def __call__(self, y, mask, inference_mode):
        for layer in self.layers:
            y = layer(y, mask, inference_mode)
        return y

    def backward(self, upstream_grad):
        # iterate through the layers in reverse order
        for layer in reversed(self.layers):
            upstream_grad = layer.backward(upstream_grad)
        return upstream_grad

    def optimize(self, optimizer, iter_num, optimizer_args={}):
        for layer in self.layers:
            layer.optimize(optimizer, iter_num, optimizer_args)
