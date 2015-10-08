import numpy
import theano

import theano.tensor as T

import activations

sigmoid = activations.get('sigmoid')
tanh = activations.get('tanh')


class Layer(object):
    def __init__(self,
                 n_in,
                 n_out,
                 previous_layer=None,
                 layer_number=1,
                 seed=123,
                 auto_setup=True,
                 dtype=theano.config.floatX):
        self.n_in = n_in
        self.n_out = n_out
        self.previous_layer = previous_layer
        self.layer_number = layer_number
        self.seed = seed
        self.dtype = dtype

        if auto_setup:
            self.init_params(seed)

    def init_params(self, seed=123):
        raise NotImplementedError

    def set_previous_layer(self, previous):
        self.previous_layer = previous

    def set_layer_number(self, number):
        self.layer_number = number

    def get_output_size(self):
        return self.n_out

    def get_input_size(self):
        return self.n_in

    def get_mask(self):
        raise NotImplementedError

    def activate(self, x):
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError

    def set_weights(self, parameters, layer_number):
        raise NotImplementedError
