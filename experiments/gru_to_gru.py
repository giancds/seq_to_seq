"""

Script to demonstrate the use of the current code version to run the (re-scaled) experiment
    described in "Sequence to Sequence Learning with Neural Networks" paper by Sutskever et al.
    (2014).

    http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf

This script used GRU units instead of LSTM in the hidden layers.

"""
import time
from seq_to_seq.embedding_layers import Embedding
from seq_to_seq.forward_layers import Softmax
from seq_to_seq.models import SequenceToSequence
from seq_to_seq.optimization import Adadelta
from seq_to_seq.recurrent_layers import GRU

from seq_to_seq.utils import DatasetIterator, load_dictionary

# some information to be used for training and validation
dict_file = '/home/gian/datasets/dict.sort.'
train_file = '/home/gian/datasets/fapesp/fapesp-v2.tok.train.'
valid_file = '/home/gian/datasets/fapesp/fapesp-v2.tok.dev.'
source_lang = 'en'
target_lang = 'pt'

en_v_size = 1000
pt_v_size = 1000
dim_proj = 10

batch_size = 128
n_epochs = 5

seed = 1234

en_dict = load_dictionary(dict_file + source_lang, max_words=en_v_size)
pt_dict = load_dictionary(dict_file + target_lang, max_words=pt_v_size)

print 'Initializing...'
time1 = time.time()

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
gru1 = GRU(dim_proj, dim_proj, seed=seed)
gru2 = GRU(dim_proj, dim_proj, seed=seed)
gru3 = GRU(dim_proj, dim_proj, return_sequences=False, seed=seed)
encoder = [emb1, gru1, gru2, gru3]

# define the decoder architecture
emb2 = Embedding(pt_v_size, dim_proj, seed=seed)
gru4 = GRU(dim_proj, dim_proj, seed=seed)
gru5 = GRU(dim_proj, dim_proj, seed=seed)
gru6 = GRU(dim_proj, dim_proj, seed=seed)
decoder = [emb2, gru4, gru5, gru6]

softmax = Softmax(dim_proj, pt_v_size)

# ensemble the sequence-to-sequence model
seq = SequenceToSequence(encoder=encoder,
                         decoder=decoder,
                         output=softmax,
                         source_v_size=en_v_size,
                         target_v_size=pt_v_size)

# set optimizer
optimizer = Adadelta()

# set up the model
seq.setup(batch_size=batch_size, optimizer=optimizer)

time2 = time.time()
print 'Initialization took %3.5f seconds. \n' % (time2 - time1)

model_file = '/home/gian/%s_%shid_prj%s_en%s_pt%s_%s_batch%s.hp5y' % \
             (gru1.__class__.__name__, len(encoder), dim_proj,
              en_v_size, pt_v_size, optimizer.__class__.__name__,
              batch_size)

print model_file

# perform the optimization
seq.train(train_data,
          valid_data,
          n_epochs=n_epochs,
          n_train_samples=160975,
          n_valid_samples=1375,
          print_train_info=True,
          save_model=True,
          filepath=model_file,
          keep_old_models=True)

print 'Done!'
