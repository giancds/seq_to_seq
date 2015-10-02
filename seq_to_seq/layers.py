import numpy
import theano

import theano.tensor as T

import activations


sigmoid = activations.get('sigmoid')
tanh = activations.get('tanh')


class LSTM(object):

    def __init__(self,
                 n_in,
                 n_out,
                 previous_layer=None,
                 return_sequences=False,
                 layer_number=1,
                 seed=123,
                 dtype=theano.config.floatX):

        self.n_in = n_in
        self.n_out = n_out
        self.previous_layer = previous_layer
        self.layer_number = layer_number
        self.seed = seed
        self.dtype = dtype

        self.return_sequences = return_sequences

        self.initial_state = None
        self.reset_initial_state = True

        self.W = None
        self.R = None
        self.b = None

        self.init_params(seed)

    def init_params(self, seed=123):

        rng = numpy.random.RandomState(seed)

        weights = rng.uniform(low=-.08, high=.08, size=(self.n_in, self.n_out*4))
        recurrent = rng.uniform(low=-.08, high=.08, size=(self.n_out, self.n_out*4))
        bias = rng.uniform(low=-.08, high=.08, size=(self.n_out*4))

        self.W = theano.shared(value=weights, name='W_%s' % self.layer_number, borrow=True)
        self.R = theano.shared(value=recurrent, name='R_%s' % self.layer_number, borrow=True)
        self.b = theano.shared(value=bias, name='b_%s' % self.layer_number, borrow=True)

    def get_layer_parameters(self):
        return [self.W, self.R, self.b]

    def set_previous_layer(self, previous):
        self.previous_layer = previous

    def set_layer_number(self, number):
        self.layer_number = number

    def get_output_size(self):
        return self.n_out

    def set_initial_state(self, initial_state):
        self.initial_state = initial_state
        self.reset_initial_state = False

    def activate(self, x, mask=None, one_step=False, initial_state=None):

        if self.previous_layer is None:
            act0 = x
        else:
            if isinstance(self.previous_layer, LSTM):
                act0 = self.previous_layer.activate(
                    x, mask, one_step, initial_state
                )
            else:
                act0 = self.previous_layer.activate(x)

        activation = self._activate(act0, mask, one_step, initial_state)
        return activation

    def _activate(self, x, mask=None, one_step=False, initial_state=None):

        if mask is None:
            mask = T.alloc(1., x.shape[0], 1)

        # input to block is (batch, time, input)
        # we want it to be  (time, batch, input)
        x = x.dimshuffle((1, 0, 2))

        xs = T.dot(x, self.W) + self.b

        if self.reset_initial_state:
            initial_state = T.alloc(
                numpy.asarray(0., dtype=self.dtype),
                x.shape[1], self.n_out
            )
        else:
            initial_state = self.initial_state

        initial_memory = T.alloc(
            numpy.asarray(0., dtype=self.dtype),
            x.shape[1], self.n_out
        )

        (hidden_state, cell), memory = theano.scan(
            self._step,
            sequences=[xs, mask],
            outputs_info=[initial_state, initial_memory],
            non_sequences=[self.R]
        )

        if self.return_sequences:
            return hidden_state.dimshuffle((1, 0, 2))
        else:
            return hidden_state[0]

    def _step(self,
              xs, m_,
              y_, c_,  # related with the recursion
              rec_weights):  # weights related to the recursion
        y_prev = m_ * y_
        c_prev = m_ * c_

        recursion = xs + T.dot(y_prev, rec_weights)

        z_hat, i_hat, f_hat, o_hat = self._slice(recursion)

        # the input become the 'candidate' information to set the memory cell
        z = tanh(z_hat)

        # compute what information from input is relevant to update memory
        i = sigmoid(i_hat)

        # compute what we should keep and what we should forget on the memory
        f = sigmoid(f_hat)

        # cell
        # z * i - scales how much of the new information is important
        # c_prev * f - forget things in memory
        # c is new cell's state based on new information and what is forgotten
        c = i * z + f * c_prev

        # output gate - (W_o * X) + b_o is pre-computed outside scan
        # compute what information from input is relevant to be forwarded
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


class Embedding(object):

    def __init__(self,
                 size,
                 dim_proj,
                 previous_layer=None,
                 layer_number=1,
                 seed=123,
                 dtype=theano.config.floatX):

        self.size = size
        self.dim_proj = dim_proj
        self.previous_layer = previous_layer
        self.layer_number = layer_number
        self.seed = seed
        self.dtype = dtype

        self.W = None

        self.init_params(seed)

    def init_params(self, seed=123):

        rng = numpy.random.RandomState(seed)

        emb = rng.uniform(low=-.1, high=.1, size=(self.size, self.dim_proj))

        self.W = theano.shared(value=emb, name='W_%s' % self.layer_number, borrow=True)

    def get_layer_parameters(self):
        return [self.W]

    def set_previous_layer(self, previous):
        self.previous_layer = previous

    def set_layer_number(self, number):
        self.layer_number = number

    def get_output_size(self):
        return self.n_out

    def activate(self, x):

        if self.previous_layer is None:
            act0 = x
        else:
            act0 = self.previous_layer.activate(x)

        activation = self.W[act0]
        return activation


class Softmax(object):

    def __init__(self,
                 n_in,
                 n_out,
                 previous_layer=None,
                 layer_number=1,
                 seed=123,
                 dtype=theano.config.floatX):

        self.n_in = n_in
        self.n_out = n_out
        self.previous_layer = previous_layer
        self.layer_number = layer_number
        self.seed = seed
        self.dtype = dtype

        self.W = None
        self.b = None

        self.init_params(seed)

    def init_params(self, seed=123):

        rng = numpy.random.RandomState(seed)

        weights = rng.uniform(low=-.01, high=.01, size=(self.n_in, self.n_out))
        # weights = numpy.zeros((self.n_in, self.n_out))
        # bias = rng.uniform(low=-.08, high=.08, size=(self.n_out))
        bias = numpy.zeros(self.n_out)

        self.W = theano.shared(value=weights, name='W_%s' % self.layer_number, borrow=True)
        self.b = theano.shared(value=bias, name='b_%s' % self.layer_number, borrow=True)

    def get_layer_parameters(self):
        return [self.W, self.b]

    def set_previous_layer(self, previous):
        self.previous_layer = previous

    def set_layer_number(self, number):
        self.layer_number = number

    def get_output_size(self):
        return self.n_out

    def activate(self, x):

        if self.previous_layer is None:
            act0 = x
        else:
            act0 = self.previous_layer.activate(x)

        dot = T.dot(act0, self.W) + self.b
        return T.nnet.softmax(dot)
