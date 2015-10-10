import numpy
import theano
import theano.tensor as T

from seq_to_seq.layers_core import Layer
from seq_to_seq.recurrent_layers import RecurrentLayer


class Softmax(Layer):

    def __init__(self,
                 n_in,
                 n_out,
                 previous_layer=None,
                 layer_number=1,
                 seed=123,
                 auto_setup=True,
                 dtype=theano.config.floatX):

        self.W = None
        self.b = None

        Layer.__init__(self,
                       n_in,
                       n_out,
                       previous_layer=previous_layer,
                       layer_number=layer_number,
                       seed=seed,
                       auto_setup=auto_setup,
                       dtype=dtype)

    def init_params(self, seed=123):

        rng = numpy.random.RandomState(seed)

        n_rows = self.n_in
        n_cols = self.n_out

        self.W = theano.shared(
            value=rng.uniform(low=-.08, high=.08, size=(n_rows, n_cols)).astype(self.dtype),
            name='W_%s' % self.layer_number, borrow=True, allow_downcast=True)

        self.b = theano.shared(
            value=rng.uniform(low=-.08, high=.08, size=(n_cols,)).astype(self.dtype),
            name='b_%s' % self.layer_number, borrow=True, allow_downcast=True)

    def get_mask(self):
        raise None

    def activate(self, x):

        if x.ndim == 3:
            shape = x.shape
            x = x.reshape([shape[0]*shape[1], shape[2]])

        if self.previous_layer is None:
            act0 = x

        else:
            act0 = self.previous_layer.activate(x)

        activation = self._activate(act0)
        return activation

    def _activate(self, x):

        dot = T.dot(x, self.W) + self.b
        act = T.nnet.softmax(dot)

        return act

    def get_weights(self):

        weights = [self.W.get_value(borrow=True),
                   self.b.get_value(borrow=True)]

        return weights

    def set_weights(self, parameters, layer_number):

        assert len(parameters) == 2, 'Wrong number of parameters to be set to Softmax layer!'

        self.layer_number = layer_number
        weights = parameters[0].value
        bias = parameters[1].value

        self.W = theano.shared(value=weights, name='W_%s' % self.layer_number, borrow=True,
                               allow_downcast=True)

        self.b = theano.shared(value=bias, name='b_%s' % self.layer_number, borrow=True,
                               allow_downcast=True)

    def get_layer_parameters(self):
        return [self.W, self.b]
