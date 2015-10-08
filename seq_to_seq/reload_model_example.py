import numpy

from seq_to_seq.layers_core import Embedding, LSTM
from seq_to_seq.models import SequenceToSequence

from utils import prepare_data, load_dictionary, load_and_convert_corpora

en_v_size = 100
pt_v_size = 100
dim_proj = 10
en_eos_idx = en_v_size-1
pt_eos_idx = pt_v_size-1

seed = 1234

# define the encoder architecture
emb1 = Embedding(en_v_size, dim_proj, seed=seed)
lstm1 = LSTM(dim_proj, dim_proj, seed=seed)
lstm2 = LSTM(dim_proj, dim_proj, return_sequences=False, seed=seed)
encoder = [emb1, lstm1, lstm2]

# define the decoder architecture
emb2 = Embedding(pt_v_size, dim_proj, seed=seed)
lstm3 = LSTM(dim_proj, dim_proj, seed=seed)
lstm4 = LSTM(dim_proj, pt_v_size, seed=seed)
decoder = [emb2, lstm3, lstm4]

# ensemble the sequence-to-sequence model
seq = SequenceToSequence(encoder=encoder,
                         decoder=decoder,
                         source_v_size=en_v_size,
                         target_v_size=pt_v_size,
                         auto_setup=False)  # set auto_setup to false to avoid initialization
                         # (weights will be overwritten anyway)

# load source and target language dictionaries
sr_dict = load_dictionary('/home/gian/datasets/dict.sort.en', max_words=en_v_size)
tr_dict = load_dictionary('/home/gian/datasets/dict.sort.pt', max_words=pt_v_size)

# load the corpora and convert its words to their indexes (corpora must be already tokenized)
sequences1 = load_and_convert_corpora('/home/gian/datasets/fapesp/fapesp-v2.tok.test-a.en', sr_dict)
sequences2 = load_and_convert_corpora('/home/gian/datasets/fapesp/fapesp-v2.tok.test-a.pt', tr_dict)

# prepare the data (add padding values to the end of each sequence so they have the same size)

seq.load_weights('/home/gian/seq_to_seq.hp5y')

seq.train(sequences1,
          sequences2,
          n_epochs=2,
          print_train_info=True)
#
# example = numpy.asarray(sequences1[0])
# seq.translate(example.reshape(1, example.shape[0]))

print 'Done!'
