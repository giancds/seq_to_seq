import numpy
import theano
import theano.tensor as T

from seq_to_seq import activations
from seq_to_seq.layers_core import Layer

sigmoid = activations.get('sigmoid')
tanh = activations.get('tanh')


class RecurrentLayer(Layer):

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

        Layer.__init__(self,
                       n_in,
                       n_out,
                       previous_layer=previous_layer,
                       layer_number=layer_number,
                       seed=seed,
                       auto_setup=auto_setup,
                       dtype=dtype)

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
        raise NotImplementedError


class LSTM(RecurrentLayer):
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

        self.W = None
        self.R = None
        self.b = None

        self.initial_state = None
        self.reset_initial_state = True

        RecurrentLayer.__init__(self,
                                n_in,
                                n_out,
                                previous_layer=previous_layer,
                                return_sequences=return_sequences,
                                return_hidden_states=return_hidden_states,
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


class GRU(RecurrentLayer):

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

        self.W = None
        self.U_i = None
        self.U_z = None
        self.U_r = None
        self.b = None

        self.initial_state = None
        self.reset_initial_state = True

        RecurrentLayer.__init__(self,
                                n_in,
                                n_out,
                                previous_layer=previous_layer,
                                return_sequences=return_sequences,
                                return_hidden_states=return_hidden_states,
                                layer_number=layer_number,
                                seed=seed,
                                auto_setup=auto_setup,
                                dtype=dtype)

    def init_params(self, seed=123):

        rng = numpy.random.RandomState(seed)

        weights = rng.uniform(low=-.08, high=.08, size=(self.n_in, self.n_out * 4))
        r_i = rng.uniform(low=-.08, high=.08, size=(self.n_out, self.n_out))
        r_z = rng.uniform(low=-.08, high=.08, size=(self.n_out, self.n_out))
        r_r = rng.uniform(low=-.08, high=.08, size=(self.n_out, self.n_out))
        bias = rng.uniform(low=-.08, high=.08, size=(self.n_out * 4))

        self.W = theano.shared(value=weights, name='W_%s' % self.layer_number, borrow=True)
        self.U_i = theano.shared(value=r_i, name='U_i_%s' % self.layer_number, borrow=True)
        self.U_z = theano.shared(value=r_z, name='U_z_%s' % self.layer_number, borrow=True)
        self.U_r = theano.shared(value=r_r, name='U_r_%s' % self.layer_number, borrow=True)
        self.b = theano.shared(value=bias, name='b_%s' % self.layer_number, borrow=True)

    def get_layer_parameters(self):
        return [self.W, self.U_i, self.U_z, self.U_r, self.b]

    def set_initial_state(self, initial_state):
        self.initial_state = initial_state
        self.reset_initial_state = False

    def _activate(self, x):

        mask = self.get_padded_shuffled_mask(x)
        # input to block is (batch, time, input)
        # we want it to be  (time, batch, input)
        x = x.dimshuffle((1, 0, 2))

        xs = T.dot(x, self.W) + self.b
        xi, xz, xr = self._slice(xs)

        if self.reset_initial_state:
            initial_state = T.unbroadcast(T.alloc(
                numpy.asarray(0., dtype=self.dtype),
                x.shape[1], self.n_out
            ))
        else:
            initial_state = self.initial_state

        state, updates = theano.scan(
                self._step,
                sequences=[xi, xz, xr, mask],
                outputs_info=[initial_state],
                non_sequences=[self.U_i, self.U_z, self.U_r],
                n_steps=x.shape[0]  # keep track of number of steps to return all computations
        )

        if self.return_sequences:
            return state.dimshuffle((1, 0, 2))
        else:
            return state[-1]

    def _step(self,
              xi, xz, xr, m_,
              s_,
              ui, uz, ur):
        s_prev = s_ * m_  # applying mask
        zi = sigmoid(xz + T.dot(s_prev, uz))  # update gate
        ri = sigmoid(xr + T.dot(s_prev, ur))  # reset gate
        s_hat = tanh(xi + T.dot((ri * s_prev), ui))  # proposed new state
        si = zi * s_hat + (1 - zi) * s_prev  # new state
        return si

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
            return m[:, :, 0 * n:1 * n], m[:, :, 1 * n:2 * n], m[:, :, 2 * n:3 * n]
        else:
            return m[:, 0 * n:1 * n], m[:, 1 * n:2 * n], m[:, 2 * n:3 * n]

    def get_weights(self):

        weights = [self.W.get_value(borrow=True),
                   self.U_i.get_value(borrow=True),
                   self.U_z.get_value(borrow=True),
                   self.U_r.get_value(borrow=True),
                   self.b.get_value(borrow=True)]

        return weights

    def set_weights(self, parameters, layer_number):

        assert len(parameters) == 5, 'Wrong number of parameters to be set to GRU layer!'

        self.layer_number = layer_number
        weights = parameters[0].value
        recs_i = parameters[1].value
        recs_z = parameters[2].value
        recs_r = parameters[3].value
        bias = parameters[4].value

        self.W = theano.shared(value=weights, name='W_%s' % self.layer_number, borrow=True)
        self.U_i = theano.shared(value=recs_i, name='U_i_%s' % self.layer_number, borrow=True)
        self.U_z = theano.shared(value=recs_z, name='U_z_%s' % self.layer_number, borrow=True)
        self.U_r = theano.shared(value=recs_r, name='U_r_%s' % self.layer_number, borrow=True)
        self.b = theano.shared(value=bias, name='b_%s' % self.layer_number, borrow=True)
