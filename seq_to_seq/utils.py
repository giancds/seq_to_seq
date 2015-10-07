import codecs
import numpy
import theano
import theano.tensor as T


def prepare_data(seqs_x, seqs_y, maxlen=None):

    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    lengths = lengths_x + lengths_y

    nb_samples = len(seqs_x)
    if maxlen is None:
        maxlen = numpy.max(lengths)

    x_padded = pad_sequences(seqs_x, maxlen, nb_samples)
    y_padded = pad_sequences(seqs_y, maxlen, nb_samples)

    # x_mask = (x_pad > -1).astype('int8')
    # y_mask = (y_pad > -1).astype('int8')

    x_padded = numpy.asarray(x_padded)
    y_padded = numpy.asarray(y_padded)
    #
    # x_mask = numpy.asarray(x_mask)
    # y_mask = numpy.asarray(y_mask)

    # return x_pad, x_mask, y_pad, y_mask
    return x_padded, y_padded


def pad_sequences(sequences, maxlen, nb_samples, dtype='int32', value=-1):

    x = (numpy.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        trunc = s[:maxlen]

        x[idx, :len(trunc)] = trunc

    return x


def generate_input_mask(x, pad=1):

    mask = T.ones_like(x.sum(axis=-1))  # is there a better way to do this without a sum?

    #  mask is (nb_samples, time)
    mask = T.shape_padright(mask)  # (nb_samples, time, 1)
    mask = T.addbroadcast(mask, -1)  # (time, nb_samples, 1) matrix.
    mask = mask.dimshuffle(1, 0, 2)  # (time, nb_samples, 1)

    if pad > 0:
        # left-pad in time with 0
        # padding = alloc_zeros_matrix(pad, mask.shape[1], 1)
        # T.alloc(np.cast[theano.config.floatX](0.), *dims)
        padding = T.alloc(numpy.cast[theano.config.floatX](0.), pad, mask.shape[1], 1)
        mask = T.concatenate([padding, mask], axis=0)
    return mask.astype('int8')


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

        3. The last index is used for the <EOS> (end-of-sentece) symbol.

    Parameters:
    -----------

        filename : string
            Name of the file containing the dictionary

        encoding : string
            File encoding. Defaults to 'utf_8'.

        skip : int
            Number of lines that should be skiped starting on the first line.

        max_words int
            Maximum number of words to have their indexes loaded. All the other words will have
                their indexes set to 0. Defaults to 50,000

    Returns:
    --------
        d : dictionary
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
                word = s[0]
                idx = index_counter
                # if index > max. number of words, set it to 0
                if line_counter < max_words:
                    d[word] = idx
                    index_counter += 1
    return d


def word_to_index(tokens, dictionary):
    """
    Utility function to convert a list of tokens into their indexes.

    Parameters:
    -----------

        tokens : list of stings
            List of strings (tokens) representing the words that should be converted to indexes.

        dictionary : python dictionary
            Dictionary with words as keys and indexes as values. To be used to convert words into
                indexes.

    Returns:
    --------
        indexes : list of int
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

    Parameters:

        filename : string
            Source file.

        dictionary : python dictionary
            Dictionary with words as keys and indexes as values. To be used to convert words into
                indexes.

        destination : string
            Destination file. Defaults to 'word_indexes.txt'.

        src_encoding : string
            Encoding of source file. Defaults to 'utf_8'.

        dest_encoding : string
            Encoding of destination file. Defaults to 'utf_8'.

        add_eos : boolean
            Flag indicating whether or not to add the <EOS> (end-of-sentence) at the end of each
                sentence prior to retrieving their indexes.

    Retuns:
    -------

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


def load_indexed_bilingual_corpora(source_language_file, target_language_file,
                                   src_encoding='utf_8', dest_encoding='utf_8'):

    source_corpora = []

    with codecs.open(source_language_file, 'r', encoding=src_encoding) as f:

        for line in f.readlines():
            tokens = line.split()
            line_tokens = []

            for token in tokens:
                line_tokens.append(int(token))

            array = numpy.asarray(line_tokens)
            source_corpora.append(array)

    target_corpora = []

    with codecs.open(target_language_file, 'r', encoding=dest_encoding) as f:

        for line in f.readlines():
            tokens = line.split()
            line_tokens = []

            for token in tokens:
                line_tokens.append(int(token))

            array = numpy.asarray(line_tokens, dtype='int8')
            target_corpora.append(array)

    return source_corpora, target_corpora
