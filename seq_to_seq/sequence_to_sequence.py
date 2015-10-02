import numpy

from seq_to_seq.layers import Embedding, LSTM, Softmax
from seq_to_seq.encoders import Encoder, Decoder

en_v_size = 100
pt_v_size = 120
dim_proj = 1000
en_eos_idx = en_v_size-1
pt_eos_idx = pt_v_size-1

seed = 1234

emb1 = Embedding(en_v_size, dim_proj,seed=seed)
lstm1 = LSTM(dim_proj, dim_proj, return_sequences=True, seed=seed)
lstm2 = LSTM(dim_proj, dim_proj, seed=seed)

emb2 = Embedding(dim_proj, dim_proj, seed=seed)
lstm3 = LSTM(dim_proj, dim_proj, return_sequences=True, seed=seed)
lstm4 = LSTM(dim_proj, dim_proj, seed=seed)
softmax = Softmax(dim_proj, pt_v_size)

encoder = Encoder(input_layer=emb1, hidden_layers=[lstm1], output_layer=lstm2)
decoder = Decoder(input_layer=emb2, hidden_layers=[lstm3, lstm4], output_layer=softmax)

# produce a toy sentence
sequence1 = numpy.random.randint(0, high=en_v_size, size=(1, 8))
sequence2 = numpy.random.randint(0, high=pt_v_size, size=(1, 8))

# encode the sequence
encoded_sequence = encoder.get_output(sequence1)

print decoder.probability_of_y_given_x(sequence2, encoded_sequence, pt_eos_idx)


print decoder.beam_search(v=encoded_sequence)

# TODO: re-implement the decode step as a beam search over all possible words


