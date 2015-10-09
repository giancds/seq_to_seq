import heapq
import h5py
import numpy
import os
import sys
import theano
import theano.tensor as T
import time

from seq_to_seq import objectives, optimization, utils


class SequenceToSequence(object):

    def __init__(self,
                 encoder,
                 decoder,
                 source_v_size=100000,
                 target_v_size=100000,
                 auto_setup=False):

        self.time1 = time.time()

        self.encoder = encoder
        self.decoder = decoder

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

        # self.output_layer.set_previous_layer(previous)
        # self.output_layer.set_layer_number(ln)

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

    def get_layers(self):
        return self.encoder + self.decoder

    def setup(self, batch_size=128):
        print '\nI\'m setting up the model now...\n'
        self.batch_size = batch_size
        self._setup_train()
        self._setup_translate()

        time2 = time.time()

        print 'Model initialization took %3.5f seconds\n' % (time2 - self.time1)

    def _setup_train(self, optimizer=None):

        if optimizer is None:
            optimizer = optimization.SGD(lr_rate=.7)

        source = T.imatrix('S')
        target = T.imatrix('T')

        encoded_source = self.encoder[-1].activate(source)

        decoder_first_hidden = self.decoder[1]  # index 1 because 0 is the embedding layer
        decoder_first_hidden.set_initial_state(encoded_source)
        decoded = self.decoder[-1].activate(target)

        # reshape the decoded vectors so it is possible to apply softmax and keep the dimensions
        shape = decoded.shape
        probs = T.nnet.softmax(decoded.reshape([shape[0]*shape[1], shape[2]]))

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

        for g in gradients:  # for all gradients
            g /= self.batch_size  # divide it by the size of the minibatch
            s = g.norm(l_norm)  # compute its norm
            if T.ge(s, threshold):  # if the norm is greater than the threshold
                g = (threshold * g) / s  # replace gradient

        return gradients

    def train(self,
              train_data,
              valid_data=None,
              n_epochs=10,
              n_train_samples=-1,
              n_valid_samples=-1,
              print_train_info=False,
              save_model=True,
              keep_old_models=False,
              filepath='sequence_to_sequence_model.hp5y',
              overwrite=True):

        train_time_1 = time.time()

        train_data.set_batch_size(self.batch_size)
        valid_data.set_batch_size(self.batch_size)

        n_train_batches = n_train_samples / self.batch_size if n_train_samples > -1 else 0
        n_valid_batches = n_valid_samples / self.batch_size if n_valid_samples > -1 else 0

        for epoch in xrange(n_epochs):

            total_loss = 0.

            # get epoch start time
            epoch_time_1 = time.time()

            # perform the minibatches and accumulate the total loss
            total_loss += self._perform_minibatches(self.train_fn,
                                                    train_data,
                                                    epoch,
                                                    n_samples=n_train_samples,
                                                    n_batches=n_train_batches,
                                                    print_train_info=print_train_info)

            valid_loss = self._evaluate_epoch(self.validate_fn,
                                              valid_data,
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

    def _perform_minibatches(self, train_fn, train_data, epoch, n_samples, n_batches,
                             print_train_info=False):

        print '\nEpoch %i \nI am performing minibatches now...' % (epoch + 1)

        total_loss = 0
        for minibatch_index in xrange(n_batches):
            time1 = time.time()
            train_x, train_y = train_data.next()
            minibatch_avg_cost = train_fn(train_x, train_y)
            total_loss += minibatch_avg_cost
            time2 = time.time()
            if print_train_info is True:
                self._print_train_info(
                    'Examples %i/%i - '
                    'Avg. loss: %.8f - '
                    'Time per batch: %3.5f' %
                    ((minibatch_index + 1) * self.batch_size, (self.batch_size * n_batches),
                     total_loss / (minibatch_index + 1),
                     (time2 - time1)))
        return total_loss

    def _evaluate_epoch(self, eval_fn, valid_data, n_batches):
        """
        Function to get the total loss (cost) of the network parameters at a given
            epoch.

        Parameters:
        ----------
            eval_fn : theano.function
                Function that will execute the loss computation.

            n_batches : int
                The number of batches to execute. Depends on the dataset size and the
                    number of batches.

            eval_type : string
                String indicating the type of the evaluation to help printing
                (i.e., it is applyed to test or validation set). Defaults to 'Test'.
        """
        loss = 0

        for i in xrange(n_batches):
            x, y = valid_data.next()
            loss += eval_fn(x, y)
        # loss = [eval_fn(i) for i in xrange(n_batches)]
        mean_loss = loss / n_batches
        print '\nValidation loss: %.8f ' % mean_loss
        return mean_loss

    def _slice_batch_data(self, x_data, y_data, minibatch_idx):

        batch_size = self.batch_size

        x = x_data[minibatch_idx * batch_size: (minibatch_idx + 1) * batch_size]

        y = y_data[minibatch_idx * batch_size: (minibatch_idx + 1) * batch_size]

        x, y = utils.prepare_data(x, y)

        return x, y

    def _print_train_info(self, info):
        """
        Help function to print train information.

        Parameters:
        ----------
            info : string
                Information to be printed

        Returns:
        --------
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

        Parameters:
        -----------

            filepath : string
                The file to be used to save the model's parameters.

            overwrite : boolean
                A flag indicating if we allow the function to overwrite an existing file. Defaults to
                    False.

        Returns:
        --------

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

        Parameters:
        -----------

            filepath : string
                The file to be used to load the parameters.

        Returns:
        --------

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
