import theano


class Layer(object):
    """
    Base class for the layers.

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
        """
        Set the previous layer in the computational path.

        :param previous: Layer class
            Object representing the previous layer in the computational path.

        :return:

        """
        self.previous_layer = previous

    def set_layer_number(self, number):
        """
        Set the number of the current layer in the computational path.

        :param number : int
            The number of the layer in the computational path.

        :return:

        """
        self.layer_number = number

    def get_output_size(self):
        """
        Set the number of the current layer in the computational path.

        :param n_out : int
            The size of the layer's output.

        :return:

        """
        return self.n_out

    def get_input_size(self):
        """
        Set the number of the current layer in the computational path.

        :param n_in : int
            The size of the layer's input.

        :return:

        """
        return self.n_in

    def get_mask(self):
        raise NotImplementedError

    def activate(self, x):
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError

    def set_weights(self, parameters, layer_number):
        raise NotImplementedError

    def get_layer_parameters(self):
        raise NotImplementedError
