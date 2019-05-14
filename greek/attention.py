from attention_decoder import *
from keras.models import Sequential
from keras.layers import LSTM
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Flatten
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
import pickle
import numpy as np

path = '/home/q078011/external/dev/greek_dev/model/pickled_models/dev/'
with open(path + 'lemma.pkl', 'rb') as file:
    lemma = pickle.load(file)


max_len_of_word = 14


all_w = [list(word) for word in lemma[0]]
to_tok = [" ".join(a) for a in all_w]

tok = Tokenizer()
tok.fit_on_texts(to_tok)
sequences = tok.texts_to_sequences(to_tok)
sequences_matrix_in = sequence.pad_sequences(sequences, maxlen=max_len_of_word)



all_w = [list(word) for word in lemma[2]]
to_tok = [" ".join(a) for a in all_w]

sequences = tok.texts_to_sequences(to_tok)
sequences_matrix_target = sequence.pad_sequences(sequences, maxlen=max_len_of_word)

with open('tok.pkl', 'wb') as h:
    pickle.dump(tok, h)

# define model

model = Sequential()
model.add(LSTM(150, input_shape=(14, 44), return_sequences=True))
model.add(LSTM(150, go_backwards=True, return_sequences=True))
model.add(AttentionDecoder(150, 44))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(to_categorical(sequences_matrix_in), to_categorical(sequences_matrix_target), validation_split=0.1, epochs=14)

model.save('att.h5')
quit()
"""
model = Sequential()
model.add(LSTM(150, input_shape=(16, 44), return_sequences=True))
model.add(LSTM(150, go_backwards=False, return_sequences=True))
model.add(AttentionDecoder(150, 44))
model.add(Flatten())
model.add(Dense(len(encoded[0]), activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['categorical_accuracy'])
model.fit(to_categorical(sequences_matrix), encoded, batch_size=size_of_batch, epochs=nb_of_epochs,
          validation_split=valid_split)
"""


from keras.models import load_model
model = load_model('att.h5', custom_objects={'AttentionDecoder': AttentionDecoder})

### todo try with one tag

"""
j = 78988
d  = {v: k for k, v in dict(tok.word_index.items()).items()}


def lem(target):
    sequences = tok.texts_to_sequences([target])
    sequences_matrix_target = sequence.pad_sequences(sequences, maxlen=max_len_of_word)
    pred = model.predict(to_categorical(sequences_matrix_target, num_classes = 44).reshape(1,16,44))
    maxes = [np.argmax(pred[0][i]) for i in range(16)]
    res = [d.get(i) for i in maxes]
    res = "".join([i for i in res if i is not None])
    print(res)
    #print([np.max(pred[0][i]) for i in range(16)])


all_w = [list(word) for word in to.split()]
to_tok = [" ".join(a) for a in all_w]
[lem(target) for target in to_tok]


# train LSTM
for epoch in range(5000):
	X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
	model.fit(X, y, epochs=1, verbose=2)
# evaluate LSTM
total, correct = 100, 0
for _ in range(total):
    X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
    yhat = model.predict(X, verbose=0)
    if array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0])):
        correct += 1

print('Accuracy: ' + str(correct))
# spot check some examples
for _ in range(100):
    X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
    yhat = model.predict(X, verbose=0)
    print('Expected:', one_hot_decode(y[0]), 'Predicted', one_hot_decode(yhat[0]))

 """
