from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer

import pandas as pd
from copy import deepcopy
import pickle
import numpy as np
import os
import itertools

from greek_accentuation.characters import *
from nltk.tag import tnt

from model.pos import get_tags, Pos, Sentence, Word
from model.clean import *

path_to_save_model = '/home/q078011/external/dev/greek_dev/model/pickled_models/dev/'
source_txt = '/home/q078011/cltk_data/greek/text/perseus_treebank_dev/v2.1/Greek/texts/'
max_len_of_word = 16
size_of_batch = 80
nb_of_epochs = 10
valid_split = 0.1
drop_rare_pos = 20
dedup = True

trainer = []
for r, d, f in os.walk(source_txt):
    trainer.append(f)

text = '\n\n'.join([get_tags(source_txt + tr) for tr in trainer[0]])

list_of_tags = Pos(text).get_words()
list_of_tags.sort()
to_train = list(list_of_tags for list_of_tags,_ in itertools.groupby(list_of_tags))


all_wrd = np.asarray(Pos(text).get_words())[:, 0]
target_pos = np.asarray(Pos(text).get_words())[:, 1]

tnt_tot = tnt.TnT()
tnt_tot.train([list(zip(list(all_wrd), list(target_pos)))])

with open(path_to_save_model + 'tnt.pkl', 'wb') as h:
    pickle.dump(tnt_tot, h)



if dedup:
    all_wrd = np.asarray(to_train)[:, 0]
    target_pos = np.asarray(to_train)[:, 1]



### filter POS to keep only value that appear more than 10 times
all_string = [x for x in target_pos]
string_df = pd.DataFrame(all_string)
string_df.columns = ['x']
nb_wrd_by_pos = string_df.groupby('x')['x'].count()
keys1 = list(nb_wrd_by_pos[nb_wrd_by_pos > drop_rare_pos].index)
values1 = deepcopy(keys1)
keys2 = list(nb_wrd_by_pos[nb_wrd_by_pos <= drop_rare_pos].index)
values2 = ['u--------'] * len(keys2)
keys1.extend(keys2)
values1.extend(values2)
simplify_dict = dict(zip(keys1, values1))

all_string = [simplify_dict.get(a) for a in all_string]
keys = list(dict.fromkeys(all_string))
values = list(range(len(keys)))
dict_g = dict(zip(keys, values))
all_w = all_string
tokens = [dict_g.get(p) for p in all_w]
encoded = to_categorical(tokens)


all_w = [list(word) for word in all_wrd]
to_tok = [" ".join(a) for a in all_w]

tok = Tokenizer()
tok.fit_on_texts(to_tok)
sequences = tok.texts_to_sequences(to_tok)
sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len_of_word)

### neural net params
alpha = to_categorical(sequences_matrix)[1].shape[1]

inputs = Input(name='inputs', shape=[max_len_of_word, alpha])
layer = LSTM(64)(inputs)
layer = Dense(256, name='FC1')(layer)
layer = Activation('relu')(layer)
layer = Dropout(0.5)(layer)
layer = Dense(len(encoded[0]), name='out_layer')(layer)
layer = Activation('softmax')(layer)
model = Model(inputs=inputs, outputs=layer)
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['categorical_accuracy'])
model.fit(to_categorical(sequences_matrix), encoded, batch_size=size_of_batch, epochs=nb_of_epochs,
          validation_split=valid_split)

model.save(path_to_save_model + 'pos_mini.h5')

with open(path_to_save_model + 'dict_letters.pkl', 'wb') as f:
    pickle.dump(dict_g, f)

with open(path_to_save_model + 'tokenizer.pkl', 'wb') as g:
    pickle.dump(tok, g)
