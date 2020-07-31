from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import os, sys

from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam, SGD
from keras.preprocessing.text import Tokenizer

try:
  import keras.backend as K
  if len(K.tensorflow_backend._get_available_gpus()) > 0:
    from keras.layers import CuDNNLSTM as LSTM
    from keras.layers import CuDNNGRU as GRU
except:
  pass


# some config
BATCH_SIZE = 128  # Batch size for training.
EPOCHS = 2000  # Number of epochs to train for.
LATENT_DIM = 25  # Latent dimensionality of the encoding space.
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE=3000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2



input_texts = []
target_texts = []
for line in open("C:/Users/PatilPra/Desktop/Prashant/Projects/rnn/seq2seq/New.txt"):
  line = line.rstrip()
  if not line:
    continue

  input_line = '<sos> ' + line
  target_line = line + ' <eos>'

  input_texts.append(input_line)
  target_texts.append(target_line)


all_lines = input_texts + target_texts

# convert the sentences (strings) into integers
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, filters='')
tokenizer.fit_on_texts(all_lines)
input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

# find max seq length
max_sequence_length_from_data = max(len(s) for s in input_sequences)
print('Max sequence length:', max_sequence_length_from_data)


# get word -> integer mapping
word2idx = tokenizer.word_index
print('Found %s unique tokens.' % len(word2idx))
assert('<sos>' in word2idx)
assert('<eos>' in word2idx)


# pad sequences so that we get a N x T matrix
max_sequence_length = min(max_sequence_length_from_data, MAX_SEQUENCE_LENGTH)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_sequence_length, padding='post')
print('Shape of data tensor:', input_sequences.shape)



# load in pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
#with open(os.path.join('C:/Users/PatilPra/Desktop/Prashant/Projects/rnn/seq2seq/glove.6B.%sd.txt' % EMBEDDING_DIM)) as f:
  # is just a space-separated text file in the format:
  # word vec[0] vec[1] vec[2] ...
f = open(('C:/Users/PatilPra/Desktop/Prashant/Projects/rnn/seq2seq/glove.6B.50d.txt' ),encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vec
    
print('Found %s word vectors.' % len(word2vec))
print("a:",word2vec['a'])


print('filling pre-trained embeddings...')
num_words = min(MAX_VOCAB_SIZE,len(word2idx)+1)
embedding_matrix = np.zeros((num_words,EMBEDDING_DIM))
for word, i in word2idx.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
one_hot_targets = np.zeros((len(input_sequences),max_sequence_length,num_words))
for i ,target_sequences in enumerate(target_sequences):
    for t,word in enumerate(target_sequences):
        if word > 0:
            one_hot_targets[i,t,word] = 1
            
#load pre-trained word embeddingd into embdding layer
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights = [embedding_matrix])


#Model
input_ = Input(shape=(max_sequence_length,))
initial_h = Input(shape=(LATENT_DIM,))
print("h:",initial_h.shape)

initial_c = Input(shape=(LATENT_DIM,))
print("c",initial_c)
x = embedding_layer(input_)
print("x",x)
print("LAtent:",LATENT_DIM)
lstm = LSTM(LATENT_DIM,return_sequences=True,return_state = True)
x,_,_ = lstm(x,initial_state=[initial_h,initial_c])
print("x",x)
dense = Dense(num_words,activation='softmax')
output=dense(x)
print("output",output)

model = Model([input_,initial_h,initial_c],output)
model.compile(
    loss = 'categorical_crossentropy',
    optimizer = Adam(lr= 0.01),
    metrics=['accuracy'])

z=np.zeros((len(input_sequences),LATENT_DIM))
r = model.fit(
    [input_sequences,z,z],
    one_hot_targets,
    batch_size=BATCH_SIZE,
    epochs = EPOCHS,
    validation_split=VALIDATION_SPLIT)

plt.plot(r.history['loss'],label='loss')
plt.plot(r.history['val_loss'],label='val_loss')
plt.legend()
plt.show()


plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

input2 = Input(shape=(1,)) # we'll only input one word at a time
x = embedding_layer(input2)
x, h, c = lstm(x, initial_state=[initial_h, initial_c]) # now we need states to feed back in
output2 = dense(x)
sampling_model = Model([input2, initial_h, initial_c], [output2, h, c])

idx2word = {v:k for k, v in word2idx.items()}


def sample_line():
  # initial inputs
  np_input = np.array([[ word2idx['<sos>'] ]])
  h = np.zeros((1, LATENT_DIM))
  c = np.zeros((1, LATENT_DIM))

  # so we know when to quit
  eos = word2idx['<eos>']

  # store the output here
  output_sentence = []

  for _ in range(max_sequence_length):
    o, h, c = sampling_model.predict([np_input, h, c])

    # print("o.shape:", o.shape, o[0,0,:10])
    # idx = np.argmax(o[0,0])
    probs = o[0,0]
    if np.argmax(probs) == 0:
      print("wtf")
    probs[0] = 0
    probs /= probs.sum()
    idx = np.random.choice(len(probs), p=probs)
    if idx == eos:
      break

    # accuulate output
    output_sentence.append(idx2word.get(idx, '<WTF %s>' % idx))

    # make the next input into model
    np_input[0,0] = idx

  return ' '.join(output_sentence)

# generate a 4 line poem
while True:
  for _ in range(10):
    print(sample_line())

  ans = input("---generate another? [Y/n]---")
  if ans and ans[0].lower().startswith('n'):
    break
