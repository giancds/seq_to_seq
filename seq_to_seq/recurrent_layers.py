import numpy
import theano
import theano.tensor as T

from seq_to_seq import activations
from seq_to_seq.layers_core import Layer

sigmoid = activations.get('sigmoid')
tanh = activations.get('tanh')


class RecurrentLayer(Layer):
    """
    Base class for recurrent layers.

    :param n_in: int
        The size of the input to the layer (i.e., the number of rows in the weight matrix).

    :param dim_proj: int
        The size of layer's output (i.e., the number of columns of the weight matrix and the bias
            vector). This is the size of the vector that will represent each of the inputs.

    :param previous_layer: Layer object
        The previous layer in the computational path.

    :param return_sequences: boolean
        Flag indicating whether or not to the layer should output the previous hidden states.

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
                 return_sequences=True,
                 layer_number=1,
                 seed=123,
                 auto_setup=True,
                 dtype=theano.config.floatX):

        self.return_sequences = return_sequences

        Layer.__init__(self,
                       n_in,
                       n_out,
                       previous_layer=previous_layer,
                       layer_number=layer_number,
                       seed=seed,
                       auto_setup=auto_setup,
                       dtype=dtype)

    def init_params(self, seed=123):
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError

    def set_weights(self, parameters, layer_number):
        raise NotImplementedError

    def get_mask(self):
        return None

    def get_padded_shuffled_mask(self, x, pad=1):
        """
        Function to provide the mask to the layer. The mask is used to 'prevent' some values to
            be used during computations.

        Notes:
        ------
            1. Slightly adapted tom from Keras Library:

                - https://github.com/fchollet/keras
                - http://keras.io/

            2. The mask returned by this function already include a time dimension.

        :param: x : theano.tensor
            The symbolic representation of the input to the layer.

        :param: pad : int
            Value to be used when padding the mask.

        :return mask : theano.tensor
            Symbolic representation of the mask including the time dimension.

        """
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
        """
        Compute the layer's output.

        :param x: theano.tensor
            Symbolic representation of the layer's input.

        :return: theano.tensor
            Symbolic representation of the layer's output.

        """
        if self.previous_layer is None:
            act0 = x
        else:
            act0 = self.previous_layer.activate(x)

        activation = self._activate(act0)
        return activation

    def _activate(self, x):
        raise NotImplementedError

    def get_layer_parameters(self):
        raise NotImplementedError


class LSTM(RecurrentLayer):
    """
    Long short therm memory (LSTM) class.

    Notes:
    ------
        1. Implemented following Greff's et al (2015) paper "LSTM: A Search Space Odyssey"

            Link: http://arxiv.org/abs/1503.04069

    :param n_in: int
        The size of the input to the layer (i.e., the number of rows in the weight matrix).

    :param dim_proj: int
        The size of layer's output (i.e., the number of columns of the weight matrix and the bias
            vector). This is the size of the vector that will represent each of the inputs.

    :param previous_layer: Layer object
        The previous layer in the computational path.

    :param return_sequences: boolean
        Flag indicating whether or not to the layer should output the previous hidden states.

    :param use_peepholes: boolean
        Flag indicating whether or not to the layer should use peephole connections.

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
                 return_sequences=True,
                 use_peepholes=False,
                 layer_number=1,
                 seed=123,
                 auto_setup=True,
                 dtype=theano.config.floatX):

        self.W = None
        self.R = None
        self.b = None

        self.initial_state = None
        self.reset_initial_state = True
        self.use_peepholes = use_peepholes

        self.P_i = None
        self.P_f = None
        self.P_o = None

        RecurrentLayer.__init__(self,
                                n_in,
                                n_out,
                                previous_layer=previous_layer,
                                return_sequences=return_sequences,
                                layer_number=layer_number,
                                seed=seed,
                                auto_setup=auto_setup,
                                dtype=dtype)

    def init_params(self, seed=123):
        """
        Function that will perform the parameter's initialization. For this layer it is a matrix
            of weights (n_in x n_out*4), a matrix of recurrent weights (n_out x n_out*4) and
            a bias vector (n_out*4).

        Notes:
        ------
            1. The 4 sets of weights are concatenated into one big matrix. This is done to speedup
                the computations. After computing the dot product of weights and input and adding
                 the bias, the result is then sliced into 4 parts, each one corresponding to the
                  inputs to the blok, input gate, forget gate and output gate.

            2. The same logic is applied to the recurrent weights matrix and bias vector.

            3. The peephole vectors are kept separated because their use during the computation
                have a different behavior than the other weights.

        :param seed: int
            A seed to feed the random number generator.

        :return:

        """

        rng = numpy.random.RandomState(seed)

        n_cols = self.n_out * 4

        self.W = theano.shared(
            value=rng.uniform(low=-.08, high=.08, size=(self.n_in, n_cols)).astype(self.dtype),
            name='W_%s' % self.layer_number, borrow=True, allow_downcast=True)
        self.R = theano.shared(
            value=rng.uniform(low=-.08, high=.08, size=(self.n_out, n_cols)).astype(self.dtype),
            name='R_%s' % self.layer_number, borrow=True, allow_downcast=True)
        self.b = theano.shared(
            value=rng.uniform(low=-.08, high=.08, size=n_cols).astype(self.dtype),
            name='b_%s' % self.layer_number, borrow=True, allow_downcast=True)

        if self.use_peepholes:
            self.P_i = theano.shared(
                value=rng.uniform(low=-.08, high=.08, size=n_cols).astype(self.dtype),
                name='P_i_%s' % self.layer_number, borrow=True, allow_downcast=True)

            self.P_f = theano.shared(
                value=rng.uniform(low=-.08, high=.08, size=n_cols).astype(self.dtype),
                name='P_f_%s' % self.layer_number, borrow=True, allow_downcast=True)

            self.P_o = theano.shared(
                value=rng.uniform(low=-.08, high=.08, size=n_cols).astype(self.dtype),
                name='P_o_%s' % self.layer_number, borrow=True, allow_downcast=True)

    def get_layer_parameters(self):
        """
        Function to return the layer's parameters (in this case, their symbolic representation).
            If we are using peephole connections, the list will have size 6 (weight matrix,
             recurrent weights matrix, bias vector and 3 peephole vectors). If we are not using
             peepholes, the list will have size 3 (weight matrix, recurrent weights matrix and
             bias vector).

        :return: params : list
            A list containing the layer's parameters in the form of theano.shared variables.

        """
        params = [self.W, self.R, self.b]

        if self.use_peepholes:
            params += [self.P_i, self.P_f, self.P_o]

        return params

    def set_initial_state(self, initial_state):
        """
        Set the initial state of the hidden layers. This is intended to be used when conditioning
            the input sequence to an encoded version of a different sequence.

        :param initial_state: theano.tensor
            Symbolic representation of the initial hidden state.

        :return:

        """
        self.initial_state = initial_state
        self.reset_initial_state = False

    def _activate(self, x):
        """
        Compute the actual activation of the layer.

        :param x: theano.tensor
            Symbolic representation of the layer's input.

        :return: theano.tensor
            Symbolic representation of the layer's activation. If the flag 'return_sequences'
                is set to True, the layer will return all the hidden states computed by scan.

        """
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

        if self.use_peepholes:
            (state, memory), updates = theano.scan(
                self._step_peep,
                sequences=[xz, xi, xf, xo, mask],
                outputs_info=[initial_state, initial_memory],
                non_sequences=[self.R, self.P_i, self.P_f, self.P_o],
                n_steps=x.shape[0]  # keep track of number of steps to return all computations
            )
        else:
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
        """
        Perform the computation step of the recursion when 'use_peepholes' flag is set to False.

        :param xz_: theano.tensor
            Symbolic representation of the input to the block (already performed dot product and
                bias addition).

        :param xi_: theano.tensor
            Symbolic representation of the input to the input gate (already performed dot product
                and bias addition).

        :param xf_: theano.tensor
            Symbolic representation of the input to the forget gate (already performed dot product
                and bias addition).

        :param xo_: theano.tensor
            Symbolic representation of the input to the output gate (already performed dot product
                and bias addition).

        :param m_: theano.tensor
            Symbolic representation of mask to the current item of the sequence.

        :param y_: theano.tensor
            Symbolic representation of the previous hidden state.

        :param c_: theano.tensor
            Symbolic representation of the previous cell content.

        :param rec_weights : theano.tensor
            Symbolic representation of the recurrent weights matrices (concatenated into one big
                matrix).

        :return: theano.tensor
            Symbolic representation of the new hidden state.

        :return: theano.tensor
            Symbolic representation of the new cell content.

        """
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

    def _step_peep(self,
                   xz_, xi_, xf_, xo_, m_,
                   y_, c_,  # related with the recursion
                   rec_weights,  # weights related to the recursion
                   pi, pf, po):  # and peepholes
        """
        Perform the computation step of the recursion when 'use_peepholes' flag is set to True.

        :param xz_: theano.tensor
            Symbolic representation of the input to the block (already performed dot product and
                bias addition).

        :param xi_: theano.tensor
            Symbolic representation of the input to the input gate (already performed dot product
                and bias addition).

        :param xf_: theano.tensor
            Symbolic representation of the input to the forget gate (already performed dot product
                and bias addition).

        :param xo_: theano.tensor
            Symbolic representation of the input to the output gate (already performed dot product
                and bias addition).

        :param y_: theano.tensor
            Symbolic representation of the previous hidden state.

        :param c_: theano.tensor
            Symbolic representation of the previous cell content.

        :param rec_weights : theano.tensor
            Symbolic representation of the recurrent weights matrices (concatenated into one big
                matrix).

        :param pi : theano.tensor
            Symbolic representation of the peephole connection to the input gate.

        :param pf : theano.tensor
            Symbolic representation of the peephole connection to the forget gate.

        :param po : theano.tensor
            Symbolic representation of the peephole connection to the output gate.

        :return: theano.tensor
            Symbolic representation of the new hidden state.

        :return: theano.tensor
            Symbolic representation of the new cell content.

        """
        y_prev = m_ * y_
        c_prev = m_ * c_

        pre = T.dot(y_prev, rec_weights)

        rz_, ri_, rf_, ro_ = self._slice(pre)

        # the input become the 'candidate' information to set the memory cell
        z_hat = xz_ + rz_
        z = tanh(z_hat)

        # compute what information from input is relevant to update memory
        i_hat = xi_ + ri_ + pi * c_prev
        i = sigmoid(i_hat)

        # compute what we should keep and what we should forget on the memory
        f_hat = xf_ + rf_ + pf * c_prev
        f = sigmoid(f_hat)

        # cell
        # z * i - scales how much of the new information is important
        # c_prev * f - forget things in memory
        # c is new cell's state based on new information and what is forgotten
        c = i * z + f * c_prev

        # output gate - (W_o * X) + b_o is pre-computed outside scan
        # compute what information from input is relevant to be forwarded
        o_hat = xo_ + ro_ + po * c
        o = sigmoid(o_hat)

        # block output
        # the last step uses what is relevant to forward and 'filters' the
        # memory cell's content
        y = o * tanh(c)

        return y, c

    def _slice(self, m):
        """
        Slice a matrix into 4 parts. Intended to be used when you have concatenated matrices and
            wants to pre-compute activations.

        :param m : theano.tensor
             Symbolic representation of the matrix to be sliced.

        :return: theano.tensor
            Symbolic representations of the sliced matrix (4 matrices).
        """

        n = self.n_out
        if m.ndim == 3:
            return m[:, :, 0 * n:1 * n], m[:, :, 1 * n:2 * n], \
                   m[:, :, 2 * n:3 * n], m[:, :, 3 * n:4 * n]
        else:
            return m[:, 0 * n:1 * n], m[:, 1 * n:2 * n], m[:, 2 * n:3 * n], m[:, 3 * n:4 * n]

    def get_weights(self):
        """
        Return a list containing the actual values of the of the layer's parameters. For this
            layer if the 'use_peepholes' flag is set to True, the list will have size 6. If the
            flag is set to false, the list will have size 3.

        :return: list
            A list containing the numpy.ndarrays representing the current weights of the layer.
                For this particular layer, if the the flag

        """
        weights = [self.W.get_value(borrow=True),
                   self.R.get_value(borrow=True),
                   self.b.get_value(borrow=True)]

        if self.use_peepholes:
            weights += [self.P_i, self.P_f, self.P_o]

        return weights

    def set_weights(self, parameters, layer_number):
        """
        Set the layer's parameters when loaded from a saved model.

        :param parameters: list
            A list containing the numpy.ndarrays representing the actual weights. For this
                particular layer, if the 'use_peepholes' flag is set to True, the list will have
                size 6. If the flag is set to false, the list will have size 3.

        :param layer_number: integer
            The position of the layer in the computational path. It is used to name the
                theano.shared variable.

        :return:

        """

        assert len(parameters) == 3 or len(parameters) == 6, \
            'Wrong number of parameters to be set to LSTM layer!'

        self.layer_number = layer_number
        weights = parameters[0].value
        recs = parameters[1].value
        bias = parameters[2].value

        self.W = theano.shared(value=weights, name='W_%s' % self.layer_number, borrow=True,
                               allow_downcast=True)
        self.R = theano.shared(value=recs, name='R_%s' % self.layer_number, borrow=True,
                               allow_downcast=True)
        self.b = theano.shared(value=bias, name='b_%s' % self.layer_number, borrow=True,
                               allow_downcast=True)

        if self.use_peepholes:
            pi = parameters[3].value
            pf = parameters[4].value
            po = parameters[5].value

            self.P_i = theano.shared(value=pi, name='P_i_%s' % self.layer_number, borrow=True,
                                     allow_downcast=True)
            self.P_f = theano.shared(value=pf, name='P_f_%s' % self.layer_number, borrow=True,
                                     allow_downcast=True)
            self.P_o = theano.shared(value=po, name='P_o_%s' % self.layer_number, borrow=True,
                                     allow_downcast=True)


class GRU(RecurrentLayer):
    """
    Gated Recurrent Unit (GRU) class.

    Notes:
    ------
        1. Implemented following Bahdanau's et al. (2015) paper: "Neural Machine Translation by
            Jointly Learning to Align and Translate".

            http://arxiv.org/abs/1409.0473

    Notes:
    ------
        1. Implemented following Greff's et al (2015) paper "LSTM: A Search Space Odyssey"

            Link: http://arxiv.org/abs/1503.04069

    :param n_in: int
        The size of the input to the layer (i.e., the number of rows in the weight matrix).

    :param dim_proj: int
        The size of layer's output (i.e., the number of columns of the weight matrix and the bias
            vector). This is the size of the vector that will represent each of the inputs.

    :param previous_layer: Layer object
        The previous layer in the computational path.

    :param return_sequences: boolean
        Flag indicating whether or not to the layer should output the previous hidden states.

    :param use_peepholes: boolean
        Flag indicating whether or not to the layer should use peephole connections.

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
                 return_sequences=True,
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
                                layer_number=layer_number,
                                seed=seed,
                                auto_setup=auto_setup,
                                dtype=dtype)

    def init_params(self, seed=123):
        """
        Function that will perform the parameter's initialization. For this layer it is a matrix
            of weights (n_in x n_out*3), three matrices of recurrent weights (n_out x n_out) and
            a bias vector (n_out*3).

        Notes:
        ------
            1. The 4 sets of weights are concatenated into one big matrix. This is done to speedup
                the computations. After computing the dot product of weights and input and adding
                 the bias, the result is then sliced into 4 parts, each one corresponding to the
                  inputs to the block, input gate, forget gate and output gate.

            2. The same logic is applied to the bias vector.

        :param seed: int
            A seed to feed the random number generator.

        :return:

        """

        rng = numpy.random.RandomState(seed)

        n_cols = self.n_out * 3

        self.W = theano.shared(
            value=rng.uniform(low=-.08, high=.08, size=(self.n_in, n_cols)).astype(self.dtype),
            name='W_%s' % self.layer_number, borrow=True, allow_downcast=True)
        self.U_i = theano.shared(
            value=rng.uniform(low=-.08, high=.08, size=(self.n_out, self.n_out)).astype(self.dtype),
            name='U_i_%s' % self.layer_number, borrow=True, allow_downcast=True)
        self.U_z = theano.shared(
            value=rng.uniform(low=-.08, high=.08, size=(self.n_out, self.n_out)).astype(self.dtype),
            name='U_z_%s' % self.layer_number, borrow=True, allow_downcast=True)
        self.U_r = theano.shared(
            value=rng.uniform(low=-.08, high=.08, size=(self.n_out, self.n_out)).astype(self.dtype),
            name='U_r_%s' % self.layer_number, borrow=True, allow_downcast=True)
        self.b = theano.shared(
            value=rng.uniform(low=-.08, high=.08, size=n_cols).astype(self.dtype),
            name='b_%s' % self.layer_number, borrow=True, allow_downcast=True)

    def get_layer_parameters(self):
        """
        Function to return the layer's parameters (in this case, their symbolic representation).
            for this particular layer, the list will have size 5.

        :return: params : list
            A list containing the layer's parameters in the form of theano.shared variables.

        """
        return [self.W, self.U_i, self.U_z, self.U_r, self.b]

    def set_initial_state(self, initial_state):
        """
        Set the initial state of the hidden layers. This is intended to be used when conditioning
            the input sequence to an encoded version of a different sequence.

        :param initial_state: theano.tensor
            Symbolic representation of the initial hidden state.

        :return:

        """
        self.initial_state = initial_state
        self.reset_initial_state = False

    def _activate(self, x):
        """
        Compute the actual activation of the layer.

        :param x: theano.tensor
            Symbolic representation of the layer's input.

        :return: theano.tensor
            Symbolic representation of the layer's activation. If the flag 'return_sequences'
                is set to True, the layer will return all the hidden states computed by scan.

        """
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
        """
        Perform the computation step of the recursion when 'use_peepholes' flag is set to False.

        :param xi : theano.tensor
            Symbolic representation of the input to the block (already performed dot product and
                bias addition).

        :param xz : theano.tensor
            Symbolic representation of the input to the input gate (already performed dot product
                and bias addition).

        :param xr : theano.tensor
            Symbolic representation of the input to the forget gate (already performed dot product
                and bias addition).

        :param m_: theano.tensor
            Symbolic representation of mask to the current item of the sequence.

        :param s_: theano.tensor
            Symbolic representation of the previous hidden state

        :param ui : theano.tensor
            Symbolic representation of the recurrent weights to the input to the unit.

        :param uz : theano.tensor
            Symbolic representation of the recurrent weights to the update gate.

        :param ur : theano.tensor
            Symbolic representation of the recurrent weights to the reset gate.

        :return: theano.tensor
            Symbolic representation of the new hidden state.

        """
        s_prev = s_ * m_  # applying mask
        zi = sigmoid(xz + T.dot(s_prev, uz))  # update gate
        ri = sigmoid(xr + T.dot(s_prev, ur))  # reset gate
        s_hat = tanh(xi + T.dot((ri * s_prev), ui))  # proposed new state
        si = zi * s_hat + (1 - zi) * s_prev  # new state
        return si

    def _slice(self, m):
        """
        Slice a matrix into 3 parts. Intended to be used when you have concatenated matrices and
            wants to pre-compute activations.

        Parameters:
        -----------
            m : theano.tensor
                Symbolic representation of a matrix of concatenated weights/biases.

        """
        n = self.n_out
        if m.ndim == 3:
            return m[:, :, 0 * n:1 * n], m[:, :, 1 * n:2 * n], m[:, :, 2 * n:3 * n]
        else:
            return m[:, 0 * n:1 * n], m[:, 1 * n:2 * n], m[:, 2 * n:3 * n]

    def get_weights(self):
        """
        Return a list containing the actual values of the of the layer's parameters. For this
            layer the list will have size 5.

        :return: list
            A list containing the numpy.ndarrays representing the current weights of the layer.
                For this particular layer, if the the flag

        """

        weights = [self.W.get_value(borrow=True),
                   self.U_i.get_value(borrow=True),
                   self.U_z.get_value(borrow=True),
                   self.U_r.get_value(borrow=True),
                   self.b.get_value(borrow=True)]

        return weights

    def set_weights(self, parameters, layer_number):
        """
        Set the layer's parameters when loaded from a saved model

        :param parameters: list
            A list containing the numpy.ndarrays representing the actual weights. For this
                particular layer, the size of the list is 5.

        :param layer_number: integer
            The position of the layer in the computational path. It is used to name the
                theano.shared variable.

        :return:

        """
        assert len(parameters) == 5, 'Wrong number of parameters to be set to GRU layer!'

        self.layer_number = layer_number
        weights = parameters[0].value
        recs_i = parameters[1].value
        recs_z = parameters[2].value
        recs_r = parameters[3].value
        bias = parameters[4].value

        self.W = theano.shared(value=weights, name='W_%s' % self.layer_number, borrow=True,
                               allow_downcast=True)
        self.U_i = theano.shared(value=recs_i, name='U_i_%s' % self.layer_number, borrow=True,
                                 allow_downcast=True)
        self.U_z = theano.shared(value=recs_z, name='U_z_%s' % self.layer_number, borrow=True,
                                 allow_downcast=True)
        self.U_r = theano.shared(value=recs_r, name='U_r_%s' % self.layer_number, borrow=True,
                                 allow_downcast=True)
        self.b = theano.shared(value=bias, name='b_%s' % self.layer_number, borrow=True,
                               allow_downcast=True)
