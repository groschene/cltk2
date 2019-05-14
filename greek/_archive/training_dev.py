from cltk.corpus.greek.beta_to_unicode import Replacer
from lxml import etree
from greek_accentuation.characters import *


from greek_accentuation.characters import strip_accents
from transliterate import translit
from cltk.corpus.greek.beta_to_unicode import Replacer

r = Replacer()

def g_translit(string):
    tr = translit(string,"el")
    if string[-1] == "s":
        tr = tr[:-1]
        tr = tr + r.beta_code('s')
    return tr

def basify(string):
    basic = "".join([strip_accents(x) for x in string])
    return basic




def get_tags():
    r = Replacer()
    entire_treebank = '/home/q078011/cltk_data/greek/text/perseus_treebank_dev/v2.1/Greek/texts/tlg0003.tlg001.perseus-grc1.1.tb.xml'
    with open(entire_treebank, 'r') as f:
        xml_string = f.read()
    root = etree.fromstring(xml_string)
    body = root.findall('body')[0]
    sentences = body.findall('sentence')
    sentences_list = []
    for sentence in sentences:
        words_list = sentence.findall('word')
        sentence_list = []
        for x in words_list:
            word = x.attrib
            form = word['form'].upper()
            form = r.beta_code(form)
            try:
                if form[-1] == 's':
                    form = form[:-1] + '?'
            except IndexError:
                pass
            form = form.lower()
            form = basify(form)
            form_list = [char for char in form if char not in [' ', "'", '?', '’', '[', ']']]
            form = ''.join(form_list)
            try:
                postag = word['postag']
            except:
                postag = 'x--------'
            if len(form) == 0: continue
            word_tag = '/'.join([form, postag])
            sentence_list.append(word_tag)
        sentence_str = ' '.join(sentence_list)
        sentences_list.append(sentence_str)
    treebank_training_set = '\n\n'.join(sentences_list)
    with open('greek_training_set_2.pos', 'w') as f:
        f.write(treebank_training_set)



get_tags()
with open('greek_training_set_2.pos', 'r') as f:
    a=f.read()


##TODO create POS object
## create word object

class word:
    def __init__(self, word_string):
        self.word_string = word_string
    def get_greek(self):
        return self.word_string.split("/")[0]
    def get_tag(self):
        return self.word_string.split("/")[1]

class sentence:
    def __init__(self, sentence_string):
        self.sentence_string = sentence_string
    def get_words(self):
        words = [word(s) for s in self.sentence_string.split(" ")]
        return words


class POS:
    def __init__(self, POSstring):
        self.POSstring = POSstring
        self.sentences = self.get_sentences()       
    def get_sentences(self):
        sentences = [sentence(s) for s in self.POSstring.split("\n\n")]
        return sentences
    def get_words_by_tag(self, pos, tag):
        out = []
        for _sentence in self.sentences:
            _words = _sentence.get_words()
            for _word in _words:
                if len(_word.get_tag())>0:
                   if _word.get_tag()[pos] == tag:
                       out.append([_word.get_greek(),_word.get_tag()])
        outer = out
        return outer

"""
POS
0 : fonction
1 : pers
2 : nombre
3 : temps (ao., fut., imp., pf...)
4 : mode (subj. opt. inf...)
5 : voie (map e?)
6 : genre
7 : cas
"""
import numpy as np
from keras.utils import to_categorical
from functools import reduce
all_str = np.asarray(POS(a).get_words_by_tag(8,'-'))[:,0]
y = np.asarray(POS(a).get_words_by_tag(8,'-'))[:,1]


all_string = [x[0] for x in y]
keys = list(dict.fromkeys(all_string))
values = list(range(len(keys)))
dict_g = dict(zip(keys, values))
all_w = all_string
tokens = [dict_g.get(p) for p in all_w]
encoded = to_categorical(tokens)


from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

all_w = [list(word) for word in all_str]
to_tok = [" ".join(a) for a in all_w]
tok = Tokenizer()
tok.fit_on_texts(to_tok)
sequences = tok.texts_to_sequences(to_tok)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=20)

all_w = [list(word) for word in all_str]
to_tok = [" ".join(a) for a in all_w]
tok = Tokenizer()
tok.fit_on_texts(to_tok)

def tokenize(all_str):
    all_w = [list(word) for word in all_str]
    to_tok_2 = [" ".join(a) for a in all_w]
    sequences = tok.texts_to_sequences(to_tok_2)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=20)
    return sequences_matrix

import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop

from keras.preprocessing import sequence
from keras.utils import to_categorical



max_len = 20
inputs = Input(name='inputs',shape=[max_len,48])
layer = LSTM(128)(inputs)
layer = Dense(128,name='FC1')(layer)
layer = Activation('relu')(layer)
layer = Dropout(0.5)(layer)
layer = Dense(len(encoded[0]),name='out_layer')(layer)
layer = Activation('softmax')(layer)
model = Model(inputs=inputs,outputs=layer)
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['categorical_accuracy'])
model.fit(to_categorical(sequences_matrix),encoded,batch_size=128,epochs=100,validation_split=0.1)

### TODO Predictor with append list
### enlarge traning set
### Build roots for lemmatizer
