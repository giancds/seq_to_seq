# -*- coding: utf-8 -*-
"""
Module containing objective (a.k.a. loss or cost) functions to be used with
    artificial neural networks.
"""
import theano.tensor as T
import theano

if theano.config.floatX == 'float64':
    off = 1.0e-9
else:
    off = 1.0e-7


def negative_log_likelihood(y_pred, y_true):
    """
        Negative log likelihood function, computed as the mean for each
            (mini)batch.

                mean( log ( P (y | X, W, b) ) )
                    - computed for all given  experiments

        Notes:
        ------
            1. Mainly used in multi-class problems.


        :param: y_pred : theano.tensor
            Symbolic variable representing the predicted y label, usually by
                the model.

       :param: y_true : theano.tensor
            Symbolic variable representing the true label of the data.

       :return: nll : theano.tensor (possibly a sybolic variable)
            Negative likelihood of the data, given the predicted and true labels
                of data.

        """
    nll = -T.log(y_pred[T.arange(y_true.shape[0]), y_true] + off).mean()
    return nll
