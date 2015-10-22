import codecs
import numpy
import theano
import theano.tensor as T


def prepare_data(seqs_x, seqs_y, maxlen=None):
    """
    Function to prepare the source and targets sequences by padding them so they have the same
        size.

    Notes:

        1. If the maxlen parameter is not provided, the length of the longest sequence within each
            list will be used as maxlen (i.e., maxlen_x and maxlen_y will be different).

        2. Therefore, sequences within each block will not have the same size.

    :param seqs_x: list
        List containing the source sequences.

    :param seqs_y: list
        List containing the target sequences.

    :param maxlen: int
        The maximum length of sequences within each block. Defaults to None.

    :return: x_padded : numpy.ndarray
        The source sequences padded to have the same length.

    :return: y_padded : numpy.ndarray
        The target sequences padded to have the same length.

    """

    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    nb_samples = len(seqs_x)
    if maxlen is None:
        maxlen_x = numpy.max(lengths_x)
        maxlen_y = numpy.max(lengths_y)
    else:
        maxlen_x = maxlen
        maxlen_y = maxlen

    x_padded = pad_sequences(seqs_x, maxlen_x, nb_samples)
    y_padded = pad_sequences(seqs_y, maxlen_y, nb_samples)

    x_padded = numpy.asarray(x_padded)
    y_padded = numpy.asarray(y_padded)

    return x_padded, y_padded


def pad_sequences(sequences, maxlen, nb_sequences, dtype='int32', value=-1):
    """
    Pad sequences so they have the same size.

    :param sequences: list
        List of sequences to pad.

    :param maxlen: int
        The maximum length of the sequences. (i.e., the size they will have after padding).

    :param nb_sequences: int
        Number of sequences in the list.

    :param dtype: theano.config
        The type of the sequence values. Defaults to 'int32'.

    :param value: int
        The valued to which the sequences should be padded. Defaults to -1.

    :return:

    """

    x = (numpy.ones((nb_sequences, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        trunc = s[:maxlen]

        x[idx, :len(trunc)] = trunc

    return x


def load_dictionary(filename, encoding='utf_8', skip=0, max_words=50000):
    """
    Utility function to load a dictionary from txt file. We assume the dictionary is already
        sorted by frequency (i.e., most frequent words at the top and least frequent words at
        the bottom).

    Notes:
    ------
        1. The dictionary should have the format:

            word idx

            Eg.:
                the 347918
                of 191298
                ....

        2. The 0-th index is used for words that will be parsed as 'unknown words'.

        3. The index 1 is used for the <EOS> (end-of-sentence) symbol.

    :param: filename : string
        Name of the file containing the dictionary

    :param: encoding : string
        File encoding. Defaults to 'utf_8'.

    :param: skip : int
        Number of lines that should be skiped starting on the first line.

    :param: max_words : int
        Maximum number of words to have their indexes loaded. All the other words will have
            their indexes set to 0. Defaults to 50,000

    :return: d : dictionary
        The loaded dictionary

    """

    d = dict()
    with codecs.open(filename, 'r', encoding=encoding) as f:
        line_counter = 0
        index_counter = 1  # we use 1 for the <EOS> symbol in both languages and 0 for <UNK> words

        d['<EOS>'] = index_counter
        index_counter += 1

        for line in f.readlines():

            line_counter += 1

            # check if we have to skip something
            if line_counter > skip:
                # split the line
                s = line.split()
                # get word and its index
                # if index > max. number of words, set it to 0
                if index_counter < max_words:
                    word = s[0]
                    d[word] = index_counter
                    index_counter += 1
    return d


def word_to_index(tokens, dictionary):
    """
    Utility function to convert a list of tokens into their indexes.

    :param: tokens : list of stings
            List of strings (tokens) representing the words that should be converted to indexes.

    :param: dictionary : python dictionary
        Dictionary with words as keys and indexes as values. To be used to convert words into
            indexes.

    :return: indexes : list of int
        The list of indexes corresponding to the words.

    """
    indexes = []
    for token in tokens:
        idx = 0
        try:
            idx = dictionary[token.lower()]
        except KeyError:
            idx = 0
        indexes.append(idx)
    return indexes


def convert_file(filename, dictionary, destination='word_indexes.txt', src_encoding='utf_8',
                 dest_encoding='utf_8', add_eos=True):
    """
    utility function to convert a tokenized file into its words indexes.

    :param: filename : string
        Source file.

    :param: dictionary : python dictionary
        Dictionary with words as keys and indexes as values. To be used to convert words into
            indexes.

    :param: destination : string
        Destination file. Defaults to 'word_indexes.txt'.

    :param: src_encoding : string
        Encoding of source file. Defaults to 'utf_8'.

    :param: dest_encoding : string
        Encoding of destination file. Defaults to 'utf_8'.

    :param: add_eos : boolean
        Flag indicating whether or not to add the <EOS> (end-of-sentence) at the end of each
            sentence prior to retrieving their indexes.

    :return:

    """
    f2 = codecs.open(destination, 'w', encoding=dest_encoding)

    try:
        with codecs.open(filename, 'r', encoding=src_encoding) as f1:
            for lines in f1.readlines():
                tokens = lines.split()
                if add_eos:
                    tokens.append('<EOS>')
                indexes = word_to_index(tokens, dictionary)
                str_indxs = ' '.join(str(i) for i in indexes)
                f2.write(str_indxs.strip())
                f2.write('\n')

        f2.close()

    except IOError:
        f2.close()

    print 'Done!'


def load_and_convert_corpora(filename, dictionary, encoding='utf_8', add_eos=True):
    """
    Load and convert a file containing a text corpus to have its words convert into indexes.

    :param filename: string
        The name of the file to be converted.

    :param dictionary: python dictionary
        A python dictionary mapping words (keys) to ints (values).

    :param encoding: string
        The file encodeing. Defaults tom 'utf_8'.

    :param add_eos: boolean
        A flag indicating whether or not to add the '<EOS>' symbol to the end of each sentence.
            Defaults to True.

    :return:

    """

    sequences = []

    try:
        with codecs.open(filename, 'r', encoding=encoding) as f1:
            for lines in f1.readlines():
                tokens = ['<EOS>'] + lines.split()
                if add_eos:
                    tokens.append('<EOS>')
                indexes = word_to_index(tokens, dictionary)
                seq = map(int, indexes)
                sequences.append(seq)

    except IOError:
        raise RuntimeError

    return sequences


class DatasetIterator(object):
    """
    Class that will perform the iteration over the dataset.

    :param: source : string
        Name of the file containing the source corpus.

    :param: target: string
        Name of the file containing the target corpus.

    :param: source_dict: python dictionary
        A python dictionary mapping words (keys) to ints (values). Corresponds to the source
            language dictionary. Defaults to None.

    :param: target_dict: python dictionary
        A python dictionary mapping words (keys) to ints (values). Corresponds to the target
            language dictionary. Defaults to None.

    :param: batch_size : int
        Size of (mini)batch. This is used to return only the amount of date to be used in one
            (mini) batch. Defaults to 128 (i.e., at each iteration it will return 128 examples).

    :param: maxlen : int
        The maximum length of each example.

    :param: source_encoding: string
        The encoding of the source corpus file. Defaults to 'utf_8'.

    :param: target_encoding: string
        The encoding of the target corpus file. Defaults to 'utf_8'.

    """
    def __init__(self,
                 source,
                 target,
                 source_dict=None,
                 target_dict=None,
                 batch_size=128,
                 maxlen=None,
                 source_encoding='utf_8',
                 target_encoding='utf_8',
                 revert_source=False):

        assert source_dict is not None
        assert target_dict is not None

        self.source = codecs.open(source, 'r', encoding=source_encoding)
        self.target = codecs.open(target, 'r', encoding=target_encoding)
        self.source_dict = source_dict
        self.target_dict = target_dict

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.revert_source = revert_source

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        """
        Reset the position of both source and target datasets to the initial position.

        """
        self.source.seek(0)
        self.target.seek(0)

    def set_batch_size(self, batch_size):
        """
        Set the (mini)batch size after we have instantiated the class.

        :param batch_size: int
            Size of the (mini)batch.

        :return:

        """
        self.batch_size = batch_size

    def next(self):
        """
        Read the next (mini)batch slice and prepare the sequences. Both source and target slices
            will have the same number of examples, defined by the class attribute 'batch_size'.

        :return: source : numpy.ndarray
            The source sequences padded to have the same length.

        :return: target : numpy.ndarray
            The target sequences padded to have the same length.

        """
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []
        ii = 0

        try:
            while True:

                ss = self.source.readline()
                tt = self.target.readline()

                if ss == "" and tt == "":
                    self.reset()
                elif ss == "" and tt != "":
                    raise IOError
                elif ss != "" and tt == "":
                    raise IOError

                ss = ss.strip().split()
                ss.append('<EOS>')
                ss = word_to_index(ss, self.source_dict)
                if self.revert_source:
                    ss = ss[::-1]

                tt = tt.strip().split()
                tt += ['<EOS>']
                tt = word_to_index(tt, self.target_dict)

                if self.maxlen is not None:
                    if len(ss) > self.maxlen and len(tt) > self.maxlen:
                        continue

                source.append(ss)
                target.append(tt)

                if len(source) >= self.batch_size or len(target) >= self.batch_size:
                    break

            source, target = prepare_data(source, target, maxlen=self.maxlen)

        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, target
