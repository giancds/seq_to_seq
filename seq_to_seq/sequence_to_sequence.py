from seq_to_seq.embedding_layers import Embedding
from seq_to_seq.models import SequenceToSequence
from seq_to_seq.recurrent_layers import LSTM

from utils import prepare_data, load_dictionary, load_and_convert_corpora

en_v_size = 100
pt_v_size = 100
dim_proj = 10
en_eos_idx = en_v_size-1
pt_eos_idx = pt_v_size-1
batch_size = 32
n_epochs = 2

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
                         target_v_size=pt_v_size)

# set up the model
seq.setup(batch_size)

# load source and target language dictionaries
sr_dict = load_dictionary('/home/gian/datasets/dict.sort.en', max_words=en_v_size)
tr_dict = load_dictionary('/home/gian/datasets/dict.sort.pt', max_words=pt_v_size)

# load the corpora and convert its words to their indexes (corpora must be already tokenized)
train_x = load_and_convert_corpora('/home/gian/datasets/fapesp/fapesp-v2.tok.test-a.en', sr_dict)
train_y = load_and_convert_corpora('/home/gian/datasets/fapesp/fapesp-v2.tok.test-a.pt', tr_dict)

valid_x = load_and_convert_corpora('/home/gian/datasets/fapesp/fapesp-v2.tok.dev.en', sr_dict)
valid_y = load_and_convert_corpora('/home/gian/datasets/fapesp/fapesp-v2.tok.dev.pt', tr_dict)

# prepare the data (add padding values to the end of each sequence so they have the same size)

seq.train(train_x,
          train_y,
          valid_x,
          valid_y,
          n_epochs=n_epochs,
          print_train_info=True,
          save_model=True,
          filepath='/home/gian/seq_to_seq.hp5y',
          keep_old_models=True)

print 'Done!'
