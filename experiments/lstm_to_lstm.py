"""

Script to demonstrate the use of the current code version to run the (re-scaled) experiment
    described in "Sequence to Sequence Learning with Neural Networks" paper by Sutskever et al.
    (2014).

    http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf

This script used GRU units instead of LSTM in the hidden layers.

"""
from seq_to_seq.embedding_layers import Embedding
from seq_to_seq.models import SequenceToSequence
from seq_to_seq.recurrent_layers import LSTM

from seq_to_seq.utils import DatasetIterator, load_dictionary

# some information to be used for training and validation
dict_file = '/home/gian/datasets/dict.sort.'
train_file = '/home/gian/datasets/fapesp/fapesp-v2.tok.test-a.'
valid_file = '/home/gian/datasets/fapesp/fapesp-v2.tok.dev.'
source_lang = 'en'
target_lang = 'pt'

en_v_size = 100
pt_v_size = 100
dim_proj = 10

batch_size = 32
n_epochs = 3

seed = 1234

en_dict = load_dictionary(dict_file + source_lang, max_words=en_v_size)
pt_dict = load_dictionary(dict_file + target_lang, max_words=pt_v_size)

# define the dataset for training
train_data = DatasetIterator(train_file + source_lang,
                             train_file + target_lang,
                             en_dict,
                             pt_dict)

# define the dataset for validation
valid_data = DatasetIterator(valid_file + source_lang,
                             valid_file + target_lang,
                             en_dict,
                             pt_dict)

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
                         target_v_size=pt_v_size)

# set up the model
seq.setup(batch_size)

# perform the optimization
seq.train(train_data,
          valid_data,
          n_epochs=n_epochs,
          n_train_samples=1314,
          n_valid_samples=1375,
          print_train_info=True,
          save_model=True,
          filepath='/home/gian/seq_to_seq.hp5y',
          keep_old_models=True)

print 'Done!'