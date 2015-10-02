import heapq
import numpy
import theano
import theano.tensor as T

from seq_to_seq import objectives, optimization


class SequenceToSequence(object):

    def __init__(self,
                 encoder,
                 decoder,
                 auto_setup=True):

        self.encoder = encoder
        self.decoder = decoder

        self._build_layer_sequence()

        self.compute_objective = None

        self.source_v_size = encoder[0].get_input_size()
        self.target_v_size = decoder[-1].get_output_size()-1

        self.encode_f = None
        self.decode_f = None

        if auto_setup:
            self.setup()

    def _build_layer_sequence(self):
        """
        Helper function to build de layer sequence.
        """
        previous = self.encoder[0]
        self.encoder[0].set_layer_number(1)

        ln = 2
        for l in xrange(len(self.encoder)):
            if l > 0:
                self.encoder[l].set_previous_layer(previous)
                self.encoder[l].set_layer_number(ln)
                previous = self.encoder[l]
            ln += 1

        previous = self.decoder[0]
        for l in xrange(len(self.decoder)):
            if l > 0:
                self.decoder[l].set_previous_layer(previous)
                self.decoder[l].set_layer_number(ln)
                previous = self.decoder[l]
            ln += 1

    def get_parameters(self):
        """

        :return:
        """
        parameters = []

        for layer in self.encoder:
            parameters += layer.get_layer_parameters()

        for layer in self.decoder:
            parameters += layer.get_layer_parameters()

        return parameters

    def setup(self, optimizer=None):
        """
        Helper function to setup the computational graph for the encoder
        """
        if optimizer is None:
            optimizer = optimization.SGD(
                lr_rate=.7,
                momentum=0.0,
                nesterov_momentum=False,
                dtype=theano.config.floatX
            )

        output_layer = self.encoder[-1]

        s = T.imatrix('S')
        v0 = output_layer.activate(s)

        self.encode_f = theano.function([s], v0, allow_input_downcast=True)

        x = T.imatrix('x')  # input to decoder
        y = T.imatrix('y')  # target sequence

        # first, set the encoder hidden state as the initial state to the decoder
        decoder_first_hidden = self.decoder[1]  # index 1 because 0 is the embedding layer
        decoder_first_hidden.set_initial_state(v0)

        # calculate the output of the network
        soft = self.decoder[-1].activate(x)

        cost = objectives.negative_log_likelihood(soft, y)
        parameters = self.get_parameters()
        backprop = optimizer.get_updates(cost, parameters)

        # compute the network output
        self.decode_f = theano.function([x, v0], soft, allow_input_downcast=True)

    def get_encoded_sequence(self, x):
        """
        Compute the hidden state of the encoder.

        Parameters:
        -----------
            x : numpy.ndarray
                The input sequence that will be encoded.

        Returns:
        --------
            encoded_sequence : numpy.ndarray
                An array representing the encoded sequence. The dimension is (1 x projection_size),
                    i.e., (1 x output_layer.n_out).

        """
        encoded_sequence = self.encode_f(x)
        return encoded_sequence

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

        Compute the probability of a sequence given a previously encoded sequence.

        Notes:
        -----
            1. Use this function to compute the probability used for the cost

        :param y:
        :param v:
        :param initial_symbol:
        :return:
        """
        if initial_symbol is None:
            initial_symbol = self.decoder[-1].get_output_size()-1

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
        Perform the beam search for decoding a given representation of an encoded sequence.
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
        """
        Compute the probability of a given set of hypotheses.
        :param hypotheses:
        :param v:
        :return:
        """

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
        """
        Extract the N-best partial hypotheses from the set of all hypotheses.

        Parameters:
        -----------

            hypotheses : numpy.ndarray
                Set of all hypotheses.

            probabilities: : numpy.ndarray
                Set containing the probability of each hypothesis in the set of all hypotheses.

            n : integer
                The number of best partial hypotheses to return (n stands for the N in N-best).

            return_probabilities : boolean
                A flag indicating whether or not to return the probability values together with
                    the N-best partial hypotheses.

        Returns:
        --------

            best_partial_hypotheses : numpy.ndarray
                The set containing only the N-best partial hypotheses.

        """
        # obtain the indexes to the highest probabilities - use heapq because it is faster than
        # sorting the entire array
        idx = heapq.nlargest(n, xrange(probabilities.shape[0]), probabilities.__getitem__)

        # extract them from the current hypotheses based on the indexes of the probabilities
        best_partial_hypothesis = hypotheses[idx]

        if return_probabilities:
            best_probabilities = probabilities[idx]
            return best_partial_hypothesis, best_probabilities

        else:
            return best_partial_hypothesis

    def _check_completed_hypotheses(self, hypotheses):
        """
        Check in the list of all hypotheses those that are considered 'completed' (i.e., they
            have <EOS> as its last symbol).

        Parameters:
        -----------

            hypotheses : numpy.ndarray
                Set of all hypotheses.

        Returns:
        --------

            complete : numpy.ndarray
                A set containing only completed hypotheses.

            idx : numpy.ndarray
                A set with the indexes of completed hypotheses.

        """
        # obtain the last column of the complete hypotheses (where are the last predicted symbols)
        last_column = hypotheses[:, -1].reshape(hypotheses.shape[0], 1)

        # get the indexes where the symbol is equal to the <EOS> symbol
        idx = numpy.where(last_column == self.target_v_size)

        # obtain the completed hypotheses
        completed = hypotheses[idx]

        return completed, idx

    def _remove_completed_hypotheses(self, hypotheses, idx):
        """
        Remove completed hypotheses from the set of all hypotheses.

        Parameters:
        -----------

            hypotheses : numpy.ndarray
                Set of all hypotheses.

            idx: : numpy.ndarray
                Set of indexes indicating which hypotheses should be removed from the set of all
                    hypotheses.

        Returns:
        --------

            cleaned : numpy.ndarray
                The set containing only incomplete hypotheses.

        """

        # remove the hypotheses according to the idx (i.e., remove the rows of the array)
        cleaned = numpy.delete(hypotheses, idx, axis=0)

        return cleaned