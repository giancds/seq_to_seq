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

        emb = rng.uniform(low=-.1, high=.1, size=(self.n_in, self.n_out))

        self.W = theano.shared(value=emb, name='W_%s' % self.layer_number, borrow=True)

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


class LSTM(Layer):
    def __init__(self,
                 n_in,
                 n_out,
                 previous_layer=None,
                 return_sequences=True,
                 return_hidden_states=False,
                 layer_number=1,
                 seed=123,
                 auto_setup=True,
                 dtype=theano.config.floatX):

        self.return_sequences = return_sequences
        self.return_hidden_states = return_hidden_states

        self.initial_state = None
        self.reset_initial_state = True

        self.W = None
        self.R = None
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

        weights = rng.uniform(low=-.08, high=.08, size=(self.n_in, self.n_out * 4))
        recurrent = rng.uniform(low=-.08, high=.08, size=(self.n_out, self.n_out * 4))
        bias = rng.uniform(low=-.08, high=.08, size=(self.n_out * 4))

        self.W = theano.shared(value=weights, name='W_%s' % self.layer_number, borrow=True)
        self.R = theano.shared(value=recurrent, name='R_%s' % self.layer_number, borrow=True)
        self.b = theano.shared(value=bias, name='b_%s' % self.layer_number, borrow=True)

    def get_layer_parameters(self):
        return [self.W, self.R, self.b]

    def set_initial_state(self, initial_state):
        self.initial_state = initial_state
        self.reset_initial_state = False

    def get_mask(self):
        return None

    def get_padded_shuffled_mask(self, x, pad=1):

        mask = self.previous_layer.get_mask()

        if mask is None:
            mask = T.ones_like(x.sum(axis=-1))  # is there a better way to do this without a sum?

        # mask is (nb_samples, time)
        mask = T.shape_padright(mask)  # (nb_samples, time, 1)
        mask = T.addbroadcast(mask, -1)  # (time, nb_samples, 1) matrix.
        mask = mask.dimshuffle(1, 0, 2)  # (time, nb_samples, 1)

        if pad > 0:
            # left-pad in time with 0
            padding = T.alloc(numpy.cast[theano.config.floatX](0.), pad, mask.shape[1], 1)
            mask = T.concatenate([padding, mask], axis=0)
        return mask.astype('int8')

    def activate(self, x):

        if self.previous_layer is None:
            act0 = x
        else:
            act0 = self.previous_layer.activate(x)

        activation = self._activate(act0)
        return activation

    def _activate(self, x):

        mask = self.get_padded_shuffled_mask(x)
        # input to block is (batch, time, input)
        # we want it to be  (time, batch, input)
        x = x.dimshuffle((1, 0, 2))

        xs = T.dot(x, self.W) + self.b
        xz, xi, xf, xo = self._slice(xs)

        if self.reset_initial_state:

            initial_state = T.unbroadcast(T.alloc(
                numpy.asarray(0., dtype=self.dtype),
                x.shape[1], self.n_out
            ))
        else:
            initial_state = self.initial_state

        initial_memory = T.unbroadcast(T.alloc(
            numpy.asarray(0., dtype=self.dtype),
            x.shape[1], self.n_out
        ))

        (state, memory), updates = theano.scan(
                self._step,
                sequences=[xz, xi, xf, xo, mask],
                outputs_info=[initial_state, initial_memory],
                non_sequences=[self.R],
                n_steps=x.shape[0]  # keep track of number of steps to return all computations
        )

        if self.return_sequences:
            return state.dimshuffle((1, 0, 2))
        else:
            return state[-1]

    def _step(self,
              xz_, xi_, xf_, xo_, m_,
              y_, c_,  # related with the recursion
              rec_weights):  # weights related to the recursion
        y_prev = m_ * y_
        c_prev = m_ * c_

        pre = T.dot(y_prev, rec_weights)

        rz_, ri_, rf_, ro_ = self._slice(pre)

        # the input become the 'candidate' information to set the memory cell
        z_hat = xz_ + rz_
        z = tanh(z_hat)

        # compute what information from input is relevant to update memory
        i_hat = xi_ + ri_
        i = sigmoid(i_hat)

        # compute what we should keep and what we should forget on the memory
        f_hat = xf_ + rf_
        f = sigmoid(f_hat)

        # cell
        # z * i - scales how much of the new information is important
        # c_prev * f - forget things in memory
        # c is new cell's state based on new information and what is forgotten
        c = i * z + f * c_prev

        # output gate - (W_o * X) + b_o is pre-computed outside scan
        # compute what information from input is relevant to be forwarded
        o_hat = xo_ + ro_
        o = sigmoid(o_hat)

        # block output
        # the last step uses what is relevant to forward and 'filters' the
        # memory cell's content
        y = o * tanh(c)

        return y, c

    def _slice(self, m):
        """
        Slice a matrix into 4 parts. Inteded to be used when you have concatenated matrices and
            wants to pre-compute activations.

        Parameters:
        -----------
            m : theano.tensor
                Symbolic representation of a mtrix of concatenated weights/biases.

        """
        n = self.n_out
        if m.ndim == 3:
            return m[:, :, 0 * n:1 * n], m[:, :, 1 * n:2 * n], \
                   m[:, :, 2 * n:3 * n], m[:, :, 3 * n:4 * n]
        else:
            return m[:, 0 * n:1 * n], m[:, 1 * n:2 * n], m[:, 2 * n:3 * n], m[:, 3 * n:4 * n]

    def get_weights(self):
        """
        Return the layer's list of parameters.

        Parameters:
        -----------

            tied_weights : boolean
                A flag indicating if the layer is sharing weights with other layers. If True, the
                    function will return only the bias. Default to False (i.e., returns both
                    weights and bias).

        Returns:
        --------
            weights : list
                A list containing either weights (pos0) and bias (pos1) or only bias (pos0).
        """
        weights = [self.W.get_value(borrow=True),
                   self.R.get_value(borrow=True),
                   self.b.get_value(borrow=True)]
        return weights

    def set_weights(self, parameters, layer_number):
        """
        This function receives the parameter list and sets to the right variables.

        Notes:
        ------
            1. It is designed to be used by the save and load functions of Networks and Encoders
                models. Need modification if want to use with different modules.

            2. The parameters is a list of h5py datasets. To get the actual values, use the
                built-in '.value' function of these datasets. Example:
                    parameters[0].value

        Parameters:
        -----------

            parameters : list of h5py datasets
                List containing the h5py datasets to be used to set the layer's weights and bias.

            layer_number : int
                The layer's position (1-based) in the computation path. Mainly used to set theano's
                    shared variables names.

        Returns:
        -------

        """
        assert len(parameters) == 3, 'Wrong number of parameters to be set to LSTM layer!'

        self.layer_number = layer_number
        weights = parameters[0].value
        recs = parameters[1].value
        bias = parameters[2].value

        self.W = theano.shared(value=weights, name='W_%s' % self.layer_number, borrow=True)
        self.R = theano.shared(value=recs, name='R_%s' % self.layer_number, borrow=True)
        self.b = theano.shared(value=bias, name='b_%s' % self.layer_number, borrow=True)
