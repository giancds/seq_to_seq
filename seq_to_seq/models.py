import numpy
import sys
import theano
import theano.tensor as T
import time

from seq_to_seq import objectives, optimization


class SequenceToSequence(object):

    def __init__(self,
                 encoder,
                 decoder,
                 source_v_size=100000,
                 target_v_size=100000,
                 auto_setup=True):

        self.encoder = encoder
        self.decoder = decoder
        # self.output_layer = output_layer

        self._build_layer_sequence()

        self.compute_objective = None

        self.source_v_size = source_v_size
        self.target_v_size = target_v_size

        self.encode_f = None
        self.train_fn = None
        self.validate_fn = None

        self.W = None
        self.b = None

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

    def setup(self, optimizer=None, seed=123):

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
        cost = cost.sum(0)
        cost = cost.mean()

        gradients = optimizer.get_gradients(cost, self.get_parameters())
        gradients = self._apply_hard_constraint_on_gradients(gradients)

        backprop = optimizer.get_updates(gradients, self.get_parameters())

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
            updates=backprop
        )

        # function to get the cost using training and test sets
        self.validate_fn = theano.function(
            inputs=[source, target], outputs=cost
        )

    def _apply_hard_constraint_on_gradients(self, gradients, threshold=5, batch=128, norm=2):

        for g in gradients:
            g_div = g / batch
            s = g_div.norm(norm)
            if T.ge(s, threshold):
                g = (threshold * g) / s

        return gradients

    def train(self,
              train_set_x,
              train_set_y,
              valid_set_x=None,
              valid_set_y=None,
              test_set_x=None,
              test_set_y=None,
              batch_size=100,
              n_epochs=10,
              print_train_info=False,
              seed=123):

        # get the number of samples on each dataset
        n_train_samples = train_set_x.shape[0]
        n_valid_samples = valid_set_x.shape[0] if valid_set_x is not None else 0
        n_test_samples = test_set_x.shape[0] if test_set_x is not None else 0

        n_train_batches = n_train_samples / batch_size
        n_valid_batches = n_valid_samples / batch_size
        n_test_batches = n_test_samples / batch_size

        total_loss = 0.

        for epoch in xrange(n_epochs):

            # shuffle the train data and labels
            numpy.random.seed(seed+epoch)
            numpy.random.shuffle(train_set_x)
            numpy.random.seed(seed+epoch)
            numpy.random.shuffle(train_set_y)

            # get epoch start time
            epoch_time_1 = time.time()

            # perform the minibatches and accumulate the total loss
            total_loss += self._perform_minibatches(self.train_fn,
                                                    train_set_x,
                                                    train_set_y,
                                                    epoch,
                                                    n_samples=n_train_samples,
                                                    n_batches=n_train_batches,
                                                    batch_size=batch_size,
                                                    print_train_info=print_train_info)
            # get epoch end tme
            epoch_time_2 = time.time()

            # print info
            print '\nEpoch %i elapsed time %3.5f' % (epoch + 1, (epoch_time_2 - epoch_time_1))
            print '\nEpoch %i averaged loss %3.10f\n' % (epoch + 1, (total_loss / n_train_batches))

    def _perform_minibatches(self, train_fn, train_set_x, train_set_y, epoch, n_samples, n_batches,
                             batch_size, print_train_info=False):

        print '\nEpoch %i \nI am performing minibatches now...' % (epoch + 1)

        total_loss = 0
        accuracy = 0
        for minibatch_index in xrange(n_batches):
            time1 = time.time()
            train_x, train_y = self._slice_batch_data(train_set_x, train_set_y,
                                                      minibatch_index, batch_size)
            # print 'Minibatch # %i' % minibatch_index
            minibatch_avg_cost = train_fn(train_x, train_y)
            total_loss += minibatch_avg_cost
            time2 = time.time()
            if print_train_info is True:
                self._print_train_info(
                    'Examples %i/%i - '
                    'Avg. loss: %.8f - '
                    'Time per batch: %3.5f' %
                    ((minibatch_index + 1) * batch_size, n_samples,
                     total_loss / (minibatch_index + 1),
                     (time2 - time1)))
        return total_loss

    def _slice_batch_data(self, x_data, y_data, minibatch_idx, batch_size):

        x = x_data[minibatch_idx * batch_size: (minibatch_idx + 1) * batch_size]

        y = y_data[minibatch_idx * batch_size: (minibatch_idx + 1) * batch_size]

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
