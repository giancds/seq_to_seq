import heapq
import h5py
import numpy
import os
import sys
import theano
import theano.tensor as T
import time

from seq_to_seq import optimization, utils


class SequenceToSequence(object):
    """
    Container to run the experiment described by Sutskever et al. (2014) in
        "Sequence to Sequence Learning with Neural Networks".

        Link:
            http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf

    Notes:
        1. If None output is provided, the model will use the last layer of the decoder to provide
            the output. This means that the parameter n_out of that layer must match the
            'target_v_size' parameter. In this case, the recurrent weights matrix of that layer
            will have the size (n_out x n_out*4). If using the default target_v_size, this means
            a matrix of 100,000 x 100,000 objects (40,000,000,000 parameters, requiring more than
            151 Gb of RAM to fit this matrix!).

    :param: encoder : list
        List of Layer objects representing the encoder portion of the model. Usually represented
            as an Embedding layer followed by one or more Recurrent layers.

    :param: decoder : list
        List of Layer objects representing the decoder portion of the model. Usually represented
            as an Embedding layer followed by one or more Recurrent layers.

    :param: output : Layer
        Layer representing the output of the model. If None is provided, used the last layer of the
            decoder to provide the output.

    :param: source_v_size : int
        Source vocabulary size. Defaults to 100,000.

    :param: target_v_size :
        target vocabulary size. Defaults to 100,000.

    :param: auto_setup : boolean
        Flag indicating whether or not the model should call the 'setup()' function.
    """

    def __init__(self,
                 encoder,
                 decoder,
                 output=None,
                 source_v_size=100000,
                 target_v_size=100000,
                 auto_setup=False):

        if output is None:
            assert decoder[-1].get_output_size() == target_v_size

        self.encoder = encoder
        self.decoder = decoder
        self.output = output

        self._build_layer_sequence()

        self.compute_objective = None

        self.source_v_size = source_v_size
        self.target_v_size = target_v_size

        self.encode_f = None
        self.train_fn = None
        self.validate_fn = None
        self.next_symbol_fn = None

        self.W = None
        self.b = None

        self.batch_size = 128

        self.auto_setup = auto_setup
        self.loaded_weights = False

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

        self.output.set_layer_number(ln)

    def get_parameters(self):
        """
        Return the symbolic representation of the parameters of all layers of the model.

        :return: list
            List of symbolic representation of the parameters of all layers.

        """
        parameters = []

        for layer in self.encoder:
            parameters += layer.get_layer_parameters()

        for layer in self.decoder:
            parameters += layer.get_layer_parameters()

        if self.output is not None:
            parameters += self.output.get_layer_parameters()

        return parameters

    def get_layers(self):
        """
        Return the layer objects of the model.

        :return: list
            List of layer objects.

        """
        if self.output is not None:
            return self.encoder + self.decoder + [self.output]
        else:
            return self.encoder + self.decoder

    def setup(self, batch_size=128, optimizer=None):
        """
        Helper function so setup the model.

        :param batch_size: int
            Size of the (mini)batch. Defaults to 128.

        :param optimizer: Optimizer object
            Optimizer object that will return the parameter updates for the model optimization.
                Defaults to None.

        :return:

        """
        print '\nI\'m setting up the model now...\n'
        self.batch_size = batch_size
        self._setup_train(optimizer)
        self._setup_translate()

    def _setup_train(self, optimizer=None):
        """
        Setup the training functions.

        :param optimizer: Optimizer object
            Optimizer object that will return the parameter updates for the model optimization.
                Defaults to None.

        :return:

        """

        if optimizer is None:
            optimizer = optimization.SGD(lr_rate=.7)

        source = T.imatrix('S')
        target = T.imatrix('T')

        encoded_source = self.encoder[-1].activate(source)

        decoder_first_hidden = self.decoder[1]  # index 1 because 0 is the embedding layer
        decoder_first_hidden.set_initial_state(encoded_source)
        decoded = self.decoder[-1].activate(target)

        # reshape the decoded vectors so it is possible to apply softmax and keep the dimensions
        if self.output is None:
            shape = decoded.shape
            probs = T.nnet.softmax(decoded.reshape([shape[0]*shape[1], shape[2]]))
        else:
            probs = self.output.activate(decoded)

        # cost
        y_flat = target.flatten()
        y_flat_idx = T.arange(y_flat.shape[0]) * self.target_v_size + y_flat
        cost = -T.log(probs.flatten()[y_flat_idx])
        cost = cost.reshape([target.shape[0], target.shape[1]])
        cost = cost.sum(axis=0)
        cost = cost.mean()

        print 'Peforming automatic differentiation...'

        gradients = optimizer.get_gradients(cost, self.get_parameters())
        gradients = self._apply_hard_constraint_on_gradients(gradients)
        #
        backprop = optimizer.get_updates(gradients, self.get_parameters())

        print 'Compiling training functions...'

        # function to get the encoded sentence
        self.encode_f = theano.function(
            [source], encoded_source, allow_input_downcast=True
        )

        # function for testing purposes
        # self.decode_f = theano.function(
        #     [target, encoded_source], prediction, allow_input_downcast=True
        # )

        # function to perform the training with parameters updates
        self.train_fn = theano.function(
            inputs=[source, target],
            outputs=cost,
            updates=backprop,
            allow_input_downcast=True
        )

        # function to get the cost using training and test sets
        self.validate_fn = theano.function(
            inputs=[source, target], outputs=cost, allow_input_downcast=True
        )

    def _setup_translate(self):
        """
        Setup the translation functions.

        :return:

        """

        # define our target
        partial_hypothesis = T.imatrix('previous_symbol')
        # define our initial state (it will change for every iteration)
        initial_state = T.matrix('initial_state')

        self.decoder[1].set_initial_state(initial_state)
        decoded = self.decoder[-1].activate(partial_hypothesis)
        decoded = decoded.dimshuffle((1, 0, 2))
        decoded = decoded[:, 0, :]  # drop the time dimension

        probs = T.nnet.softmax(decoded[-1])

        self.next_symbol_fn = theano.function([partial_hypothesis, initial_state], probs[-1],
                                              allow_input_downcast=True)

    def _apply_hard_constraint_on_gradients(self, gradients, threshold=5, l_norm=2):
        """
        Function to apply a hard constraint on the parameter's gradients.

        :param gradients: theano.tensor
            Symbolic representation of the  parameter's gradients.

        :param threshold: int
            The threshold to which apply the constraints. Defaults to 5 (i.e., if the norm exceeds
                5, the constraint is applied.

        :param l_norm: int
            The number of the norm to compute. Defaults to 2 (i.e., L2-norm).

        :return: gradients: theano.tensor
            Symbolic representation of the parameter's gradients with/without the constraint
                applied.

        """

        for g in gradients:  # for all gradients
            g /= self.batch_size  # divide it by the size of the minibatch
            s = g.norm(l_norm)  # compute its norm
            if T.ge(s, threshold):  # if the norm is greater than the threshold
                g = (threshold * g) / s  # replace gradient

        return gradients

    def train(self,
              train_data_iterator,
              valid_data_iterator=None,
              n_epochs=10,
              n_train_samples=-1,
              n_valid_samples=-1,
              evaluate_before_start=False,
              print_train_info=False,
              save_model=True,
              keep_old_models=False,
              filepath='sequence_to_sequence_model.hp5y',
              overwrite=True):
        """
        Function to be called to start the training (optimization) procedure.

        :param train_data_iterator: DatasetIterator object
            The dataset iterator corresponding to the training data.

        :param valid_data_iterator:
            The dataset iterator corresponding to the validation data. Defaults to None.

        :param n_epochs: int
            The number of epochs to train. Defaults to 10.

        :param n_train_samples: int
            Number of samples in the training data. Defaults to -1.

        :param n_valid_samples:
            Number of samples in the validation data. Defaults to -1.

        :param evaluate_before_start: boolean
            A flag indicating whether or not to compute the validation loss prior to start
                training. Defaults to False.

        :param print_train_info: boolean
            A flag indicating whether or not to print information about the training process.
                Defaults to False.

        :param save_model: boolean
            A flag indicating whether or not to save the model during/after the training.
                Defaults to True.

        :param keep_old_models: boolean
            A flag indicating whether or not to keep old models when saving the model during
                training. Defaults to False.

        :param filepath: string
            The name of the file to save the model.

        :param overwrite: boolean
            A flag indicating whether or not to overwrite old models when saving. Defaults to
                True.

        :return:
        """

        train_time_1 = time.time()

        train_data_iterator.set_batch_size(self.batch_size)
        valid_data_iterator.set_batch_size(self.batch_size)

        n_train_batches = n_train_samples / self.batch_size if n_train_samples > -1 else 0
        n_valid_batches = n_valid_samples / self.batch_size if n_valid_samples > -1 else 0

        if evaluate_before_start:
            valid_loss = self._evaluate_epoch(self.validate_fn,
                                              valid_data_iterator,
                                              n_batches=n_valid_batches)

        for epoch in xrange(n_epochs):

            total_loss = 0.

            # get epoch start time
            epoch_time_1 = time.time()

            # perform the minibatches and accumulate the total loss
            total_loss += self._perform_minibatches(self.train_fn,
                                                    train_data_iterator,
                                                    epoch,
                                                    n_batches=n_train_batches,
                                                    print_train_info=print_train_info)

            valid_loss = self._evaluate_epoch(self.validate_fn,
                                              valid_data_iterator,
                                              n_batches=n_valid_batches)

            # get epoch end tme
            epoch_time_2 = time.time()

            if save_model:
                new_file = filepath

                if keep_old_models:
                    new_file = new_file + '_epoch_' + str(epoch+1)

                self.save_weights(new_file, overwrite=overwrite)

            # print info
            print 'Epoch %i elapsed time %3.5f' % (epoch + 1, (epoch_time_2 - epoch_time_1))

        train_time_2 = time.time()

        if save_model:
            self.save_weights(filepath, overwrite=overwrite)

        print '\nTotal training time: %3.5f' % (train_time_2 - train_time_1)

    def _perform_minibatches(self, train_fn, train_data, epoch, n_batches,
                             print_train_info=False):
        """
        Function that will actually perform the (mini)batches.

        :param train_fn: theano.function
            Function that will perform one (mini)batch.

        :param train_data: DatasetIterator object
            Object that will iterate over the dataset to retrieve the (mini)batches slices.

        :param epoch: int
            Epoch number.

        :param n_batches: : int
            Number of (mini)batches in each epoch.

        :param print_train_info: boolean
            Flag indicating whether or not to print information about the training process.
                Defaults to False.

        :return: float
            Accumulated loss for the current epoch.
        """

        print '\nEpoch %i \nI am performing minibatches now...' % (epoch + 1)

        accumulated_loss = 0
        time_acc = 0
        for minibatch_index in xrange(n_batches):
            time1 = time.time()
            train_x, train_y = train_data.next()
            minibatch_avg_cost = train_fn(train_x, train_y)
            accumulated_loss += minibatch_avg_cost
            time2 = time.time()
            time_acc += time2
            if print_train_info is True:
                self._print_train_info(
                    'Examples %i/%i - '
                    'Avg. loss: %.8f - '
                    'Time per batch: %3.5f (Average: %3.5f)' %
                    ((minibatch_index + 1) * self.batch_size, (self.batch_size * n_batches),
                     accumulated_loss / (minibatch_index + 1),
                     (time2 - time1), (time_acc / n_batches)))
        return accumulated_loss

    def _evaluate_epoch(self, eval_fn, valid_data, n_batches):
        """
        Function to get the total loss (cost) of the network parameters at a given
            epoch.

        :param eval_fn : theano.function
            Function that will execute the loss computation on the validation dataset.

        :param valid_data: DatasetIterator object
            Object that will iterate over the dataset to retrieve the (mini)batches slices.

        :param n_batches : int
            The number of (mini)batches to execute.

        :return mean_loss : float
            The averaged loss computed over the validation data.

        """
        loss = 0

        for i in xrange(n_batches):
            x, y = valid_data.next()
            loss += eval_fn(x, y)
        # loss = [eval_fn(i) for i in xrange(n_batches)]
        mean_loss = loss / n_batches
        print '\nValidation loss: %.8f ' % mean_loss
        return mean_loss

    def _print_train_info(self, info):
        """
        Help function to print train information.

        :param info : string
            Information to be printed

        """
        sys.stdout.write("\r")  # ensure stuff will be printed at the same line during an epoch
        sys.stdout.write(info)
        sys.stdout.flush()

    def translate(self, source, beam_size=2, return_probabilities=False):
        encoded_sequence = self.encode_f(source)

        completed_hypotheses = []
        best_hypotheses = numpy.ones((1, 1))

        while len(completed_hypotheses) < beam_size:
            # generate a set of new hypotheses
            new_hypotheses = self._generate_new_hypotheses(best_hypotheses)

            v = numpy.tile(encoded_sequence, (new_hypotheses.shape[0], 1))

            all_probabilities = self.next_symbol_fn(new_hypotheses, v)

            # extract the N-best hypothesis
            best_hypotheses = self._extract_n_best_partial_hypothesis(
                new_hypotheses, all_probabilities, n=beam_size
            )

            # check those that are completed hypothesis
            completed, idx = self._check_completed_hypotheses(best_hypotheses)

            # transform the completed hypotheses and their probabilities into list and zip
            # them into tuples before assigning them to the set of completed hypothesis
            completed_hypotheses += completed.tolist()

            # remove the completed hypotheses from the list of best hypotheses
            best_hypotheses = self._remove_completed_hypotheses(best_hypotheses, idx)

        return completed_hypotheses

    def _generate_new_hypotheses(self, old_hypotheses=None):
        # get the output size (should be equal to the target vocabulary size)
        size = self.target_v_size

        # create a new matrix containing the actual hypothesis stacked 'size' times
        new_hypotheses = None

        for i in xrange(old_hypotheses.shape[0]):
            tiled = numpy.tile(old_hypotheses[i], (size, 1))
            if i == 0:
                new_hypotheses = tiled
            else:
                new_hypotheses = numpy.vstack((new_hypotheses, tiled))

        # generate an array containing all the symbols representing the target vocabulary
        appendix = numpy.asarray(xrange(size))
        appendix = appendix.reshape(appendix.shape[0], 1)
        appendix = numpy.tile(appendix, (old_hypotheses.shape[0], 1))

        # append the array into the matrix of hypothesis
        new_hypotheses = numpy.hstack(
            (new_hypotheses, appendix)
        )

        return new_hypotheses

    def _extract_n_best_partial_hypothesis(self, hypotheses, probabilities, n=2):

        # obtain the indexes to the highest probabilities
        idx = heapq.nlargest(n, xrange(probabilities.shape[0]), probabilities.__getitem__)

        # extract them from the current hypotheses based on the indexes of the probabilities
        best_partial_hypothesis = hypotheses[idx, :]

        return best_partial_hypothesis

    def _check_completed_hypotheses(self, hypotheses):

        last_column = hypotheses[:, -1]

        # get the indexes where the symbol is equal to the <EOS> symbol
        idx = numpy.where(last_column.reshape(last_column.shape[0], 1) == 1)

        # obtain the completed hypotheses
        completed = hypotheses[idx]

        return completed, idx

    def _remove_completed_hypotheses(self, hypotheses, idx):

        # remove the hypotheses according to the idx (i.e., remove the rows of the array)
        cleaned = numpy.delete(hypotheses, idx, axis=0)

        return cleaned

    def save_weights(self, filepath, overwrite=False):
        """
        Function to save the model's parameters.

            Slighlty adapted from Keras library.

                https://github.com/fchollet/keras

        :param filepath : string
            The file to be used to save the model's parameters.

        :param overwrite : boolean
            A flag indicating if we allow the function to overwrite an existing file. Defaults to
                False.

        """
        # Save weights from all layers to HDF5

        # if file exists and should not be overwritten
        if not overwrite and os.path.isfile(filepath):
            import sys
            get_input = input
            if sys.version_info[:2] <= (2, 7):
                get_input = raw_input
            overwrite = get_input('[WARNING] %s already exists - overwrite? [y/n]' % (filepath))
            while overwrite not in ['y', 'n']:
                overwrite = get_input('Enter "y" (overwrite) or "n" (cancel).')
            if overwrite == 'n':
                return
            print('[TIP] Next time specify overwrite=True in save_weights!')

        layers = self.get_layers()

        f = h5py.File(filepath, 'w')
        f.attrs['nb_layers'] = len(layers)
        for k, l in enumerate(layers):
            g = f.create_group('layer_{}'.format(k))
            weights = l.get_weights()
            g.attrs['nb_params'] = len(weights)
            for n, param in enumerate(weights):
                param_name = 'param_{}'.format(n)
                param_dset = g.create_dataset(param_name, param.shape, dtype=param.dtype)
                param_dset[:] = param
        f.flush()
        f.close()

    def load_weights(self, filepath):
        """
        Function to save the model's parameters.

            Slighlty adapted from Keras library.

                https://github.com/fchollet/keras

        Notes:
            1. The function assumes that there is a model architecture previously defined and in
                place to set the parameters loaded from the file. In addition, the model must have
                the same architecture.

        :param filepath : string
                The file to be used to load the parameters.

        """
        layers = self.get_layers()
        # Loads weights from HDF5 file
        f = h5py.File(filepath)
        for k in range(f.attrs['nb_layers']):
            if k > 0:
                g = f['layer_{}'.format(k)]
                weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
                layers[k].set_weights(weights, k)
        f.close()
        self.loaded_weights = True
