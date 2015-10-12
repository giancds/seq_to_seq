import numpy
import theano
import theano.tensor as T

from seq_to_seq.layers_core import Layer
from seq_to_seq import activations

softmax = activations.get('softmax')


class Softmax(Layer):
    """
    Softmax class.

    :param n_in: int
        The size of the input to the layer (i.e., the number of rows in the weight matrix).

    :param n_out: int
        The size of layer's output (i.e., the number of columns of the weight matrix and the bias
            vector). This is the size of the vector that will represent each of the inputs.

    :param previous_layer: Layer object
        The previous layer in the computational path.

    :param layer_number: int
        The layer position in the computational path.

    :param seed: int
        The seed to feed the random number generator.

    :param auto_setup: boolean
        Flag indicating if the model should call setup() when initializing the model or leave it
            to the user to call it explicitly.

    :param dtype: theano.config.floatX
        Type of floating point to be used.

    :return:
    """

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
        """
        Function that will perform the parameter's initialization. For this layer it is a matrix
            of weights (n_in x n_out) and a vector of bias (n_out).

        :param seed: int
            A seed to feed the random number generator.

        :return:

        """

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
        """
        Return the mask to be applied to the inputs. The mask is used to 'prevent' some values to
            be used during computations. This value will be passed to the next layer on the
            computational path.

        Notes:
        ------
            1. Softmax layer is supposed to be the last one in the computational path. Therefore,
                there is no other layer to receive the mask. Given this fact, we return None as
                the mask value.

        :return: None
            The representation of the mask.

        """
        raise None

    def activate(self, x):
        """
        Compute the layer's output. The output of this layer can be interpreted as a probability
            distribution over the output units.

        :param x: theano.tensor
            Symbolic representation of the layer's input.

        :return: theano.tensor
            Symbolic representation of the layer's output.

        """
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
        """
        compute the actual activation of the layer.

        :param x: theano.tensor
            Symbolic representation of the layer's input.

        :return: theano.tensor
            Symbolic representation of the layer's activation.

        """

        dot = T.dot(x, self.W) + self.b
        act = softmax(dot)

        return act

    def get_weights(self):
        """
        Return a list containing the actual values of the of the layer's parameters. For this
            layer it will be a list of length 2 (weights and bias)

        :return: list
            A list containing the numpy.ndarrays representing the current weights of the layer.

        """

        weights = [self.W.get_value(borrow=True),
                   self.b.get_value(borrow=True)]

        return weights

    def set_weights(self, parameters, layer_number):
        """
        Set the layer's parameters when loaded from a saved model

        :param parameters: list
            A list containing the numpy.ndarrays representing the actual weights. For this
                particular layer, the size of the list is 2.

        :param layer_number: integer
            The position of the layer in the computational path. It is used to name the
                theano.shared variable.

        :return:

        """

        assert len(parameters) == 2, 'Wrong number of parameters to be set to Softmax layer!'

        self.layer_number = layer_number
        weights = parameters[0].value
        bias = parameters[1].value

        self.W = theano.shared(value=weights, name='W_%s' % self.layer_number, borrow=True,
                               allow_downcast=True)

        self.b = theano.shared(value=bias, name='b_%s' % self.layer_number, borrow=True,
                               allow_downcast=True)

    def get_layer_parameters(self):
        """
        Function to return the layer's parameters (in this case, their symbolic representation).
            For this layer, it will be a list of size 2 (weight matrix and bias vector).

        :return: list
            A list containing the layer's parameters in the form of theano.shared variables. For
                this layer it is a matrix of weights (n_in x n_out) and a vector of bias (n_out).

        """
        return [self.W, self.b]
