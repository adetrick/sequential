from sequential.optimizers import *


class Layer:
    '''
    Base class for trainable layers with optimization support.

    Provides shared functionality including optimization handling 
    for layers like Dense, RNN, LSTM, and LayerNorm. This class 
    should not be used directly. Subclasses must implement their 
    own forward, backward, and build methods while inheriting 
    these optimization utilities.
    '''

    def __init__(self):
        # map trainable params in each layer with a unique id
        self.trainable_params = {}
        # save gradients for each trainable param
        # during backpropagation using the same id
        self.trainable_params_grad = {}
        # map each trainable param to an optimizer
        self.optimizers = {}

    def optimize(self, optimizer, iter_=None, **kwargs):
        '''
        updates parameters using gradients calculated 
        during backpropagation
        '''
        if not self.optimizers:
            self.init_optimizers(optimizer, **kwargs)

        for id_, grads in self.trainable_params_grad.items():
            params = self.trainable_params[id_]
            optimizer_obj = self.optimizers[id_]
            if optimizer == 'gd':
                self.trainable_params[id_] = optimizer_obj.optimize(params, grads)
            elif optimizer == 'adam':
                self.trainable_params[id_] = optimizer_obj.optimize(params, grads, iter_=iter_)

    def init_optimizers(self, optimizer, **kwargs):
        for id_ in self.trainable_params:
            if optimizer == 'gd':
                self.optimizers[id_] = GradientDescent(**kwargs)
            elif optimizer == 'adam':
                self.optimizers[id_] = AdamOptimizer(**kwargs)
