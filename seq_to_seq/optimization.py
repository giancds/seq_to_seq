# -*- coding: utf-8 -*-
"""
Module containing optimizer algorithms to train the network.
"""
import theano
import numpy
import theano.tensor as T
import logging

log = logging.getLogger(__name__)


class Optimizer(object):
    """
    Base class for optimizers that will train the models.
    """

    def __init__(self,
                 lr_rate=1e-1,
                 lr_rate_annealing=0.0,
                 anneal_start=1,
                 anneal_end=10,
                 anneal_type='step_decay',
                 dtype=theano.config.floatX):
        self.lr_rate = theano.shared(value=numpy.cast[dtype](lr_rate), name='learning_rate')
        self.lr_rate_annealing = lr_rate_annealing
        self.anneal_start = anneal_start
        self.anneal_end = anneal_end
        self.current_epoch = 0
        self.anneal_type = anneal_type
        self.dtype = dtype

    def _get_gradients(self, objective_function, parameters):
        """
        Return the gradients in symbolic format following Theano's conventions.

        Parameters:
        ----------
            objective_function : theano.tensor
                    Objective function defined by symbols.

            parameters : list if theano.shared
                Parameters (weights and bias from the model)

        Returns:
        -------
            gradients : list of theano.tensor
                The gradients with respect to the parameters supplied to the function.

        """
        gradients = [T.grad(objective_function, param) for param in parameters]
        return gradients

    def _share_zeroed_parameter(self, parameters, name, dtype=theano.config.floatX):
        """
        Function to create theano shared variables from the current model's parameters.
        """
        if len(parameters) > 1:
            shared = [theano.shared(p.get_value() * numpy.asarray(0., dtype=dtype), name=name % p)
                      for p in parameters]
        else:
            p = parameters[0]
            shared = theano.shared(p.get_value() * numpy.asarray(0., dtype=dtype), name=name % p)
        return shared

    def set_current_epoch_number(self, epoch):
        """
        Function to set the current epoch number. It will be used to perform learning rate
            annealing (mainly on SGD - Stochastic Gradient Descent).

        epoch : int
            The current epoch number.

        """
        self.current_epoch = epoch

    def get_updates(self, cost, parameters):
        """
        Abstract method to get parameter updates for the optimization process.

        """
        raise NotImplementedError


class Adadelta(Optimizer):
    """

    An adaptive learning rate optimizer.

    Notes:
    ------

        1. Based on the code by Pierre Luc Carrier and Kyunghyun Cho:

            LSTM Networks for Sentiment Analysis:

                http://deeplearning.net/tutorial/lstm.html

        2. For more information, see Zeiler's (2012) paper:

            http://arxiv.org/abs/1212.5701


    Parameters:
    ----------
        lr_rate : float
            In this implementation, lr_rate corresponds to the 'rho' parameter in Zeiler's
                formulation of adadelta. This parameter 'is a decay constant similar to that
                used in momentum method' (Zeiler, 2012).

        epsilon : float
            'This constant serves the purpose both to start off the first iteration where
                delta_x_0 = 0 and to ensure progress continues to be made even if previous
                updates become small' (Zeiler, 2012).

        dtype : theano.config
            Float type to be used. May affect speed and GPU usage.

    Notes:
    -----
        Parameters alpha, epsilon and gamma should not be altered.

        Creates several zipped lists to ensure every list has the same size and
            has the same order given the parameters.

        Creates several shared variables to be used among all the calculations
            and updates.This is necessary because Theano will update the values
            of the variables and this could cause interference on the rmsprop
            computations.

    """

    def __init__(self,
                 lr_rate=1e-6,
                 epsilon=.95,
                 dtype=theano.config.floatX):
        Optimizer.__init__(self,
                           lr_rate=lr_rate,
                           lr_rate_annealing=0.0,
                           dtype=dtype)
        self.epsilon = epsilon

    def get_updates(self, cost, parameters=None):
        """
        Method to get parameter updates for the optimization process according to RMSProp.

        The implementation used in this version is based on Alex Graves' paper:

            http://arxiv.org/abs/1308.0850

        Parameters:
        -----------
            cost : theano.tensor
                Symbolic representation of the network cost (a.k.a. loss, objective)

            parameters : list
                The list of parameters to update (possibly a list of theano.shared)

        Return:
        -------
            updates : list of tupples
                List containing tupples formed by a parameter and its update.

        """
        assert (parameters is not None), 'Passing empty parameters to Adadelta'

        if self.lr_rate.get_value() > 1e-4:
            print 'WARNING: Adadelta performs better with lr_rate = 1e-4 (0.0001)'

        if self.lr_rate_annealing is not 0.0:
            print 'WARNING: Learning rate annealing is not suported for this implementation of ' \
                  'Adadelta. I am ignoring it.'

        gradients = self._get_gradients(cost, parameters)

        # share some parameters to perform the update
        zipped_grads = self._share_zeroed_parameter(parameters, name='%s_grad', dtype=self.dtype)
        deltas = self._share_zeroed_parameter(parameters, name='%s_accs', dtype=self.dtype)
        accs = self._share_zeroed_parameter(parameters, name='%s_deltas', dtype=self.dtype)

        zgup = [(zg, g) for zg, g in zip(zipped_grads, gradients)]

        # update the 'accumulators'
        rg2up = [(rg2, self.epsilon * rg2 + (1 - self.epsilon) * (g ** 2))
                 for rg2, g in zip(accs, gradients)]

        # use current deltas to generate an update version of them
        updir = [-T.sqrt(ru2 + self.lr_rate) / T.sqrt(rg2 + self.lr_rate) * zg
                 for zg, ru2, rg2 in zip(zipped_grads, deltas, accs)]

        # use the updated version of deltas to update the old ones
        ru2up = [(ru2, self.epsilon * ru2 + (1 - self.epsilon) * (ud ** 2))
                 for ru2, ud in zip(deltas, updir)]

        # generate parameter updates
        parameter_updates = [(p, p + ud) for p, ud in zip(parameters, updir)]

        return zgup + rg2up + ru2up + parameter_updates


class SGD(Optimizer):
    """
        Class to (symbolicaly) define the parameter updates used in Stochastic Gradient Descent
            (SGD) optimization.

        Parameters:
        ----------
            lrn_rate: float
                The amount to which the parameters should updated.

            momentum : float
                Momentum constant. Deafult to 0.0. Leave it at 0.0 if don't want to use it.

            nesterov : boolean
                Whether or not to use the Nesterov's momentum rule. Default to False (i.e., apply
                    standard momentum rule).

            lr_rate_annealing : float
                The ammount by which we will anneal the learning_rate. Default to 0.0 (i.e., no
                    annealing).

            anneal_start : int
                The number of epoch to start the learning_rate annealing. Defaults to 1.

            anneal_end : int
                The number of epoch to start the learning_rate annealing. Defaults to 10.

            anneal_type : string
                The type of annealing to perform.

                    If 'step_decay', anneal is based only on lr_rate_annealing using the formula:
                        lr_rate * (1.0 / (1 + lr_rate_annealing))

                    If '1_t_decay', include also the epoch number and use the formula
                        lr_rate * (1.0 / (1 + lr_rate_annealing * current_epoch))

                Defaults to 'step_decay'.

            dtype : theano.config
                Float type to be used. May affect speed and GPU usage.

        Returns:
        -------
            a tuple of array_like:
                The update rules for the parameters of each layer according to SGD.

        """

    def __init__(self,
                 lr_rate=1e-1,
                 momentum=0.0,  # leave momentum at 0.0 if don't want to use it
                 nesterov_momentum=False,
                 lr_rate_annealing=0.0,
                 anneal_start=1,
                 anneal_end=10,
                 anneal_type='step_decay',
                 dtype=theano.config.floatX):
        Optimizer.__init__(self,
                           lr_rate=lr_rate,
                           lr_rate_annealing=lr_rate_annealing,
                           anneal_start=anneal_start,
                           anneal_end=anneal_end,
                           anneal_type=anneal_type,
                           dtype=dtype)
        self.momentum = momentum
        self.nesterov = nesterov_momentum

    def get_updates(self, cost, parameters=None):
        """
        Method to get parameter updates for the optimization process according to Stochastic
            Gradient Descent (SGD).

        Parameters:
        -----------
            cost : theano.tensor
                Symbolic representation of the network cost (a.k.a. loss, objective)

            parameters : list
                The list of parameters to update (possibly a list of theano.shared)

        Return:
        -------
            updates : list of tupples
                List containing tupples formed by a parameter and its update.

        """

        assert (parameters is not None), 'Passing empty parameters to Stochastic Gradient Descent'

        # new_lr = self.lr_rate.get_value(borrow=True)
        new_lr = self.lr_rate.get_value()
        # new_lr = self.lr_rate

        # if the current epoch is inside the annealing start/end
        # if self.anneal_start <= self.current_epoch <= self.anneal_end:
        if self.anneal_type is 'step_decay':
            new_lr = self.lr_rate * (1.0 / (1 + self.lr_rate_annealing)) \
                if self.anneal_start <= self.current_epoch <= self.anneal_end else new_lr
        elif self.anneal_type is '1_t_decay':
            new_lr = self.lr_rate * (1.0 / (1 + self.lr_rate_annealing * self.current_epoch)) \
                if self.anneal_start <= self.current_epoch <= self.anneal_end else new_lr

        self.lr_rate.set_value(numpy.cast[self.lr_rate.dtype](new_lr))
        # self.lr_rate = new_lr

        # define the gradients of parameters
        grad_theta = self._get_gradients(cost, parameters)

        # parameters' updates
        updates = []
        append = updates.append  # use this to avoid dot calls - it can give some speedup

        for param, gradient in zip(parameters, grad_theta):
            m = self._share_zeroed_parameter([param], 'momentum_%s' % param, dtype=self.dtype)
            v = self.momentum * m - new_lr * gradient
            if self.nesterov is True:
                new_p = param + self.momentum * v - new_lr * gradient
            else:
                new_p = param + v

            append((m, v))
            append((param, new_p))

        return updates


class RMSProp(Optimizer):
    """
    Alternative version of Stochastic Gradient Descent that normalize the
        gradients based on their recent magnitudes before the parameter updates.

    Notes:
    ------

        1. Based on the code by Pierre Luc Carrier and Kyunghyun Cho:

            LSTM Networks for Sentiment Analysis:

                http://deeplearning.net/tutorial/lstm.html

        2. For more details on the formulation used in this function, please refer to
            Alex Graves' paper:

            'Generating Sequences With Recurrent Neural Networks' (2012) at :
                http://arxiv.org/pdf/1308.0850v5.pdf

        3. Parameters alpha, epsilon and gamma should not be altered.

        4. Creates several zipped lists to ensure every list has the same size and
            has the same order given the parameters.

        5. Creates several shared variables to be used among all the calculations
            and updates.This is necessary because Theano will update the values
            of the variables and this could cause interference on the rmsprop
            computations.

    Parameters:
    ----------
        lr_rate : float
           The amount to which the parameters should updated.

        alpha : float
            The factor to compute the g_i and n_i parameters in Alex Grave's formula.

        beta : float
            The rate to which the gradients should decay before updating them.

        gamma : float
            Damping factor.

        dtype : theano.config
            Float type to be used. May affect speed and GPU usage.

    """

    def __init__(self,
                 lr_rate=1e-4,
                 alpha=.95,
                 beta=0.9,
                 gamma=1e-4,
                 dtype=theano.config.floatX):

        Optimizer.__init__(self,
                           lr_rate=lr_rate,
                           lr_rate_annealing=0.0,
                           dtype=dtype)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def get_updates(self, cost, parameters):
        """
        Method to get parameter updates for the optimization process according to RMSProp.

        The implementation used in this version is based on Alex Graves' paper:

            http://arxiv.org/abs/1308.0850

        Parameters:
        -----------
            cost : theano.tensor
                Symbolic representation of the network cost (a.k.a. loss, objective)

            parameters : list
                The list of parameters to update (possibly a list of theano.shared)

        Return:
        -------
            updates : list of tupples
                List containing tupples formed by a parameter and its update.

        """
        assert (parameters is not None), 'Passing empty parameters to RMSprop'

        if self.lr_rate.get_value() > 1e-4:
            print 'WARNING: RMSprop performs better with lr_rate = 1e-4 (0.0001)'

        if self.lr_rate_annealing is not 0.0:
            print 'WARNING: Learning rate annealing is not suported for this implementation of ' \
                  'RMSProp. I am ignoring it.'

        gradients = self._get_gradients(cost, parameters)

        # share some parameters to perform the update
        zipped_grads = self._share_zeroed_parameter(parameters, name='%s_grad', dtype=self.dtype)
        running_grads = self._share_zeroed_parameter(parameters, name='%s_rgrad', dtype=self.dtype)
        running_grads2 = self._share_zeroed_parameter(parameters, name='%s_grad2', dtype=self.dtype)
        updir = self._share_zeroed_parameter(parameters, name='%s_updir')

        grads = [(zg, g) for zg, g in zip(zipped_grads, gradients)]

        # g_i terms on Alex's formula
        g_i = [(rg, self.alpha * rg + (1 - self.alpha) * g)
               for rg, g in zip(running_grads, gradients)]

        # g_i terms on Alex's formula
        n_i = [(rg2, self.alpha * rg2 + (1 - self.alpha) * (g ** 2))
               for rg2, g in zip(running_grads2, gradients)]

        new_grads = [(ud, self.beta * ud - self.lr_rate * zg / T.sqrt(rg2 - rg ** 2 + self.gamma))
                     for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads, running_grads2)]

        parameter_updates = [(p, p + udn[1]) for p, udn in zip(parameters, new_grads)]

        return g_i + n_i + parameter_updates

