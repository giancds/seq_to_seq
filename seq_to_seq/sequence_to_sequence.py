import numpy

from seq_to_seq.layers import Embedding, LSTM, Softmax
from seq_to_seq.models import SequenceToSequence

en_v_size = 100
pt_v_size = 100
dim_proj = 10
en_eos_idx = en_v_size-1
pt_eos_idx = pt_v_size-1

seed = 1234

emb1 = Embedding(en_v_size, dim_proj, seed=seed)
lstm1 = LSTM(dim_proj, dim_proj, return_sequences=True, seed=seed)
lstm2 = LSTM(dim_proj, dim_proj, seed=seed)

emb2 = Embedding(pt_v_size, dim_proj, seed=seed)
lstm3 = LSTM(dim_proj, dim_proj, return_sequences=True, seed=seed)
lstm4 = LSTM(dim_proj, dim_proj, seed=seed)
softmax = Softmax(dim_proj, pt_v_size)

encoder = [emb1, lstm1, lstm2]
decoder = [emb2, lstm3, lstm4, softmax]

seq = SequenceToSequence(encoder, decoder)

# produce a toy sentence
sequence1 = numpy.random.randint(0, high=en_v_size, size=(1, 8))
sequence2 = numpy.random.randint(0, high=pt_v_size, size=(1, 8))

# encode the sequence
encoded_sequence = seq.get_encoded_sequence(sequence1)

print seq.probability_of_y_given_x(sequence2, encoded_sequence, pt_eos_idx)

print seq.beam_search(v=encoded_sequence)

# TODO: re-implement the decode step as a beam search over all possible words


