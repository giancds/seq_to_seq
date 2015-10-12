# -*- coding: utf-8 -*-
"""
Module containing activation functions to be used in layers of neural networks.
    It makes heavy use of Theano and Numpy built-in functions.

    Currently implementing:
        Hyperbolic Tangent (tanh)
        Identity (return the input without transforming it)
        REctified Linear Unit (RELU)
        Sigmoid  (logistic)
        Softmax
"""
import theano.tensor as T
import theano


def get(act='tanh'):
    """
    Function to return the correct activation function as specified by the parameters.

    :param: act : string
        Name of the activation to return.

    :return: func: function
        The activation function specified by the parameter.
    """
    func = None
    if act is 'tanh':
        func = tanh
    elif act is 'sigmoid':
        func = sigmoid
    elif act is 'softmax':
        func = softmax
    elif act is 'identity':
        func = identity
    elif act is 'relu':
        func = relu
    else:
        print 'WARNING: Activation not found. ' \
              'I am returning tahn activation by default.'
    return func


def identity(input_to_function):
    """
    Identity activation. Return the same input to the unit without any
        modifications. Usually aplied to InputLayers.

    :param: input_to_function : theano.tensor
        Symbolic matrix (or vector) corresponding to the input.

    :return: theano.tensor
        Symbolic matrix (or vector) corresponding to the input.

    """
    return input_to_function


def relu(input_to_function):
    """
    Rectified linear activation. The multiplication in the step will turn all the negative
            values to 0.

    :param: input_to_function : theano.tensor
        Symbolic matrix (or vector) corresponding to the input.

    :return: theano.tensor
         Symbolic matrix (or vector) corresponding to the activation of the
            layer.

        """
    # this funtion call will convert all the negative values from input_to_function to 0
    return T.maximum(T.cast(0., theano.config.floatX), input_to_function)


def sigmoid(input_to_function):
    """
    Sigmoid (a.k.a logistic) as defined in theano.tensor package. Apply the function element-wise.

    :param: input_to_function : theano.tensor
        Symbolic matrix (or vector) corresponding to the input.

    :return:  theano.tensor
        Symbolic matrix (or vector) corresponding to the activation of the layer.

        """
    return T.nnet.sigmoid(input_to_function)


def softmax(input_to_function):
    """
    Softmax activation as defined in theano.tensor package. Apply the function element-wise.
        Generally used as the last layer activation (output) as it can be interpreted as a
        probability distribution over the labels (i.e., the probability of the data taking a
        certain label).

    :param: input_to_function : theano.tensor
        Symbolic matrix (or vector) corresponding to the input.

    :return: theano.tensor
           Softmax activation applyed element-wise to the array_like input..

    """
    x = input_to_function
    sftmx = T.nnet.softmax(x)
    return sftmx


def tanh(input_to_function):
    """
    Hyperbolic tangent as defined in theano.tensor package. Apply the function element-wise.

    :param: input_to_function : theano.tensor
        Symbolic matrix (or vector) corresponding to the input.

    :return: theano.tensor
        Symbolic matrix (or vector) corresponding to the activation of the layer.

    """
    return T.tanh(input_to_function)
