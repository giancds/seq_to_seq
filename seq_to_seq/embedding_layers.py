import numpy
import theano
import theano.tensor as T

from seq_to_seq.layers_core import Layer


class Embedding(Layer):
    def __init__(self,
                 size,
                 dim_proj,
                 previous_layer=None,
                 layer_number=1,
                 seed=123,
                 auto_setup=True,
                 dtype=theano.config.floatX):

        self.W = None
        self.current_mask = None

        Layer.__init__(self,
                       size,
                       dim_proj,
                       previous_layer=previous_layer,
                       layer_number=layer_number,
                       seed=seed,
                       auto_setup=auto_setup,
                       dtype=dtype)

    def init_params(self, seed=123):
        rng = numpy.random.RandomState(seed)

        self.W = theano.shared(value=rng.uniform(low=-.1, high=.1, size=(self.n_in, self.n_out)),
                               name='W_%s' % self.layer_number,
                               borrow=True)

    def get_layer_parameters(self):
        return [self.W]

    def get_mask(self):
        return self.current_mask

    def activate(self, x):
        if self.previous_layer is None:
            act0 = x
        else:
            act0 = self.previous_layer.activate(x)

        activation = self.W[act0]

        self.current_mask = T.ones_like(x) * (1 - T.eq(x, -1))

        return activation

    def get_weights(self):
        weights = [self.W.get_value(borrow=True)]
        return weights

    def set_weights(self, parameters, layer_number):

        assert len(parameters) == 1, 'Wrong number of parameters to be set to EmbbedingLayer!'

        self.layer_number = layer_number
        w = parameters[0].value

        self.W = theano.shared(value=w, name='W_%s' % self.layer_number, borrow=True)
