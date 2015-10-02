import heapq
import numpy
import theano
import theano.tensor as T


class Model(object):

    def __init__(self,
                 input_layer,
                 hidden_layers=None,
                 output_layer=None,
                 auto_setup=True):

        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer

        self._build_layer_sequence()

        if auto_setup:
            self.setup()

    def setup(self):
        raise NotImplementedError

    def _build_layer_sequence(self):
        """
        Helper function to build de layer sequence.
        """
        previous = self.input_layer
        self.input_layer.set_layer_number(1)

        ln = 2
        for layer in self.hidden_layers:
            layer.set_previous_layer(previous)
            layer.set_layer_number(ln)
            previous = layer
            ln += 1

        self.output_layer.set_layer_number(ln)
        self.output_layer.set_previous_layer(previous)

    def get_output(self, x, v=None):
        raise NotImplementedError


class Encoder(Model):

    def __init__(self,
                 input_layer,
                 hidden_layers=None,
                 output_layer=None,
                 auto_setup=True):

        self.encode_f = None

        Model.__init__(self,
                       input_layer,
                       hidden_layers=hidden_layers,
                       output_layer=output_layer,
                       auto_setup=auto_setup)

    def setup(self):
        """
        Helper function to setup the computational graph for the encoder
        """
        x = T.imatrix()
        v0 = self.output_layer.activate(x)
        self.encode_f = theano.function([x], v0, allow_input_downcast=True)

    def get_output(self, x, v=None):
        """
        Compute the hidden state of the encoder.

        Parameters:
        -----------
            x : numpy.ndarray
                The input sequence that will be encoded.

            v : None
                Not actually used. Kept to match the 'super class' signature

        Returns:
        --------
            encoded_sequence : numpy.ndarray
                An array representing the encoded sequence. The dimension is (1 x projection_size),
                    i.e., (1 x output_layer.n_out).

        """
        encoded_sequence = self.encode_f(x)
        return encoded_sequence


class Decoder(Model):

    def __init__(self,
                 input_layer,
                 hidden_layers=None,
                 output_layer=None,
                 auto_setup=True):

        self.decode_f = None

        Model.__init__(self,
                       input_layer,
                       hidden_layers=hidden_layers,
                       output_layer=output_layer,
                       auto_setup=auto_setup)

        self.target_v_size = self.output_layer.get_output_size()

    def setup(self):
        """
        Helper function to setup the computational graph for the decoder.
        """
        x = T.imatrix()  # input to decoder
        v = T.matrix()  # initial state of decoder's
        s = T.matrix()  # softmax result

        # first, set the encoder hidden state as the initial state to the decoder
        first_hidden = self.hidden_layers[0]
        first_hidden.set_initial_state(v)

        # calculate the output of the network
        soft = self.output_layer.activate(x)

        # compute the network output
        self.decode_f = theano.function([x, v], soft, allow_input_downcast=True)

    def get_initial_sequence(self, eos_idx):
        """
        Generate an initial sequence to start the decoding

        Parameters:
        -----------
            eos_idx : integer
                Integer representing the index of the <EOS> symbol.

        Returns:
        --------
            initial_sequence : numpy.ndarray
                A (1 x 1) matrix containing the index of the <EOS> symbol in the target language.

        """
        # create an initial sequence
        initial_sequence = numpy.asarray([eos_idx - 1])
        initial_sequence = initial_sequence.reshape(1, 1)

        return initial_sequence

    def get_probabilities(self, x, v=None):
        """
        Return the probabilities ofa given sequence and the initial state.

        :param x:
        :param v:
        :return:
        """
        assert v is not None

        # decode the sequence x given the hidden state v
        probs = self.decode_f(x, v)
        return probs

    def probability_of_y_given_x(self, y, v, initial_symbol=None, beam_search=False):
        """

        :param y:
        :param v:
        :param initial_symbol:
        :return:
        """
        if initial_symbol is None:
            initial_symbol = self.output_layer.get_output_size()-1

        # assuming initial probability is 1 given the fact that we'll always use the <EOS> symbol
        # as the first input to the decoder
        total_prob = 1

        if not beam_search:
            # turn it into a 1 x 1 matrix and stack the initial symbol to the target sequence
            y_ = numpy.hstack((numpy.asarray([initial_symbol]).reshape(1, 1), y))
        else:
            # turn the input into a 1 x size_of_y matrix because we will append <EOS> latter
            y_ = numpy.asarray(y).reshape(1, len(y))

        # for all symbols in the stacked sequence
        for i in xrange(y_.shape[1]):
            if i > 0:  # start after the first symbol which is supposed to be the <eos>
                next_symbol = y_[0, i]    # get the symbol index
                x = numpy.asarray(y_[0, 0:i+1]).reshape(1, i+1)  # slice the sequence
                all_probabilities = self.get_probabilities(x, v)  # compute all probabilities
                symbol_prob = all_probabilities[0][next_symbol]  # extract our target probability
                total_prob *= symbol_prob   # multiply by the previous probabilities

        return total_prob

    def generate_new_hypotheses(self, old_hypothesis):
        """

        :param old_hypothesis:
        :return:
        """
        # get the output size (should be equal to the target vocabulary size)
        size = self.target_v_size

        # create a new matrix containing the actual hypothesis stacked 'size' times
        new_hypotheses = None

        for i in xrange(old_hypothesis.shape[0]):
            tiled = numpy.tile(old_hypothesis[i], (size, 1))
            if i == 0:
                new_hypotheses = tiled
            else:
                new_hypotheses = numpy.vstack((new_hypotheses, tiled))

        # generate an array containing all the symbols representing the target vocabulary
        appendix = numpy.asarray(xrange(size))
        appendix = appendix.reshape(appendix.shape[0], 1)
        appendix = numpy.tile(appendix, (old_hypothesis.shape[0], 1))

        # append the array into the matrix of hypothesis
        new_hypotheses = numpy.hstack(
            (new_hypotheses, appendix)
        )

        return new_hypotheses

    def beam_search(self, initial_sequence=None, v=None, beam_size=2, return_probabilities=True):
        """
        """
        assert v is not None

        completed_hypoteses = []

        if initial_sequence is None:
            initial_sequence = self.get_initial_sequence(self.target_v_size)

        # given the input sequence, generate a series of new hypotheses
        new_hypotheses = self.generate_new_hypotheses(initial_sequence)

        while len(completed_hypoteses) < beam_size:
            # compute all probabilities of the generated hypotheses
            all_probabilities = self._compute_hypotheses_probabilities(new_hypotheses, v)

            # extract the N-best hypothesis
            best_hypotheses, probs = self._extract_n_best_partial_hypothesis(
                new_hypotheses, all_probabilities, n=beam_size,
                return_probabilities=return_probabilities
            )

            # check those that are completed hypothesis
            completed, idx = self._check_completed_hypotheses(best_hypotheses)

            # extract the probability of completed hypotheses
            completed_probs = probs[idx]

            # transform the completed hypotheses and their probabilities into list and zip
            # them into tuples before assigning them to the set of completed hypothesis
            completed_hypoteses += zip(completed.tolist(), completed_probs.tolist())

            # remove the completed hypotheses from the list of best hypotheses
            best_hypotheses = self._remove_completed_hypotheses(best_hypotheses, idx)

            # generate a set of new hypothesis
            new_hypotheses = self.generate_new_hypotheses(best_hypotheses)

        return completed_hypoteses

    def _compute_hypotheses_probabilities(self, hypotheses, v):

        all_probabilities = numpy.ones((hypotheses.shape[0], 1))

        i = 0
        # for all generated hypotheses, get its probabilities given the
        for h in hypotheses:
            p = self.probability_of_y_given_x(h, v, beam_search=True)
            all_probabilities[i] = p
            i += 1

        return all_probabilities

    def _extract_n_best_partial_hypothesis(self, hypotheses, probabilities, n=2,
                                           return_probabilities=False):

        # obtain the indexes to the highest probabilities
        idx = heapq.nlargest(n, xrange(probabilities.shape[0]), probabilities.__getitem__)

        # extract them from the current hypotheses based on the indexes of the probabilities
        best_partial_hypothesis = hypotheses[idx]

        if return_probabilities:
            best_probabilities = probabilities[idx]
            return best_partial_hypothesis, best_probabilities

        else:
            return best_partial_hypothesis

    def _check_completed_hypotheses(self, hypotheses):

        # obtain the last column of the complete hypotheses (where are the last predicted symbols)
        last_column = hypotheses[:, -1].reshape(hypotheses.shape[0], 1)

        # get the indexes where the symbol is equal to the <EOS> symbol
        idx = numpy.where(last_column == self.target_v_size)

        # obtain the completed hypotheses
        completed = hypotheses[idx]

        return completed, idx

    def _remove_completed_hypotheses(self, hypotheses, idx):

        # remove the hypotheses according to the idx (i.e., remove the rows of the array)
        cleaned = numpy.delete(hypotheses, idx, axis=0)

        return cleaned
