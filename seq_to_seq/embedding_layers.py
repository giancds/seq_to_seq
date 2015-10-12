import numpy
import theano
import theano.tensor as T

from seq_to_seq.layers_core import Layer


class Embedding(Layer):
    """
    Embedding class.

    :param size: int
        The size of the layer (i.e., the number of rows, the size of the input vocabulary).

    :param dim_proj: int
        The size of the projection (i.e., the number of columns). This is the size of the vector
            that will represent each of the inputs.

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
        """
        Function that will perform the parameter's initialization. For this layer it is a matrix
            (size x dim_proj).

        :param seed: int
            A seed to feed the random number generator.

        :return:

        """
        rng = numpy.random.RandomState(seed)

        self.W = theano.shared(
            value=rng.uniform(low=-.1, high=.1, size=(self.n_in, self.n_out)).astype(self.dtype),
            name='W_%s' % self.layer_number, borrow=True, allow_downcast=True)

    def get_layer_parameters(self):
        """
        Function to return the layer's parameters

        :return: list
            A list containing the layer's parameters in the form of theano.shared variables. For
                this layer it is a matrix (size x dim_proj).

        """
        return [self.W]

    def get_mask(self):
        """
        Return the mask to be applied to the inputs. The mask is used to 'prevent' some values to
            be used during computations.

            Example: input = [1, 2, 3, 8, 9]  mask = [1, 1, 1, 0, 0] - if we apply 'mask' to the
                'input', the last 2 values (corresponding to 0s in the mask) will no be used when
                performing the computations.

        Notes:
        ------
            1. A new mask is computed whenever new data is passed for activation.

        :return: theano.tensor
            Symbolic representation of the mask.

        """
        return self.current_mask

    def activate(self, x):
        """
        Compute the layer's output. For this layer it turns a single index into a vector of
            (dim_proj) size.

        :param x: theano.tensor
            Symbolic representation of the layer's input.

        :return: theano.tensor
            Symbolic representation of the layer's output.

        """
        if self.previous_layer is None:
            act0 = x
        else:
            act0 = self.previous_layer.activate(x)

        activation = self.W[act0]

        self.current_mask = T.ones_like(x) * (1 - T.eq(x, -1))

        return activation

    def get_weights(self):
        """
        Return a list containing the actual values of the of the layer's parameters. For this
            layer it will be a list of length 1 (just weights).

        :return: list
            A list containing the numpy.ndarrays representing the current weights of the layer.

        """
        weights = [self.W.get_value(borrow=True)]
        return weights

    def set_weights(self, parameters, layer_number):
        """
        Set the layer's parameters when loaded from a saved model

        :param parameters: list
            A list containing the numpy.ndarrays representing the actual weights. For this
                particular layer, the size of the list is 1.

        :param layer_number: integer
            The position of the layer in the computational path. It is used to name the
                theano.shared variable.

        :return:

        """

        assert len(parameters) == 1, 'Wrong number of parameters to be set to EmbbedingLayer!'

        self.layer_number = layer_number
        w = parameters[0].value

        self.W = theano.shared(value=w, name='W_%s' % self.layer_number, borrow=True)
