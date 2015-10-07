import numpy

from seq_to_seq.layers import Embedding, LSTM
from seq_to_seq.models import SequenceToSequence

from utils import prepare_data, load_dictionary, load_and_convert_corpora

en_v_size = 100
pt_v_size = 100
dim_proj = 10
en_eos_idx = en_v_size-1
pt_eos_idx = pt_v_size-1

seed = 1234

emb1 = Embedding(en_v_size, dim_proj, seed=seed)
lstm1 = LSTM(dim_proj, dim_proj, seed=seed)
lstm2 = LSTM(dim_proj, dim_proj, return_sequences=False, seed=seed)

emb2 = Embedding(pt_v_size, dim_proj, seed=seed)
lstm3 = LSTM(dim_proj, dim_proj, seed=seed)
lstm4 = LSTM(dim_proj, pt_v_size, seed=seed)

encoder = [emb1, lstm1, lstm2]
decoder = [emb2, lstm3, lstm4]

seq = SequenceToSequence(encoder=encoder,
                         decoder=decoder,
                         source_v_size=en_v_size,
                         target_v_size=pt_v_size)


sr_dict = load_dictionary('/home/gian/datasets/dict.sort.en', max_words=en_v_size)
tr_dict = load_dictionary('/home/gian/datasets/dict.sort.pt', max_words=pt_v_size)

sequences1 = load_and_convert_corpora('/home/gian/datasets/fapesp/fapesp-v2.tok.test-a.en', sr_dict)
sequences2 = load_and_convert_corpora('/home/gian/datasets/fapesp/fapesp-v2.tok.test-a.pt', tr_dict)

# produce a toy sentence
sequence1 = [sequences1[0], sequences1[1], sequences1[2]]
sequence2 = [sequences2[0], sequences2[1], sequences2[2]]

seq1, seq2 = prepare_data(sequences1, sequences2)

# # encode the sequence
# encoded_sequence = seq.encode_f(seq1)
#
# print encoded_sequence
#
# print encoded_sequence.shape

seq.train(seq1,
          seq2,
          batch_size=100,
          n_epochs=10,
          print_train_info=True)
