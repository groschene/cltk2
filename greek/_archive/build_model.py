from cltk.corpus.greek.beta_to_unicode import Replacer
from lxml import etree
from greek_accentuation.characters import *
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from functools import reduce
from keras.preprocessing import sequence
from keras.utils import to_categorical
import numpy as np
from keras.preprocessing.text import Tokenizer
import pandas as pd
from copy import deepcopy
from external.greek_dev.clean import *
from nltk.tag import tnt
from keras.metrics import top_k_categorical_accuracy
import pickle

path = '/home/q078011/external/greek_dev/'
root = '/home/q078011/cltk_data/greek/text/perseus_treebank_dev/v2.1/Greek/texts/'

trainer = ["tlg0003.tlg001.perseus-grc1.1.tb.xml",
"tlg0011.tlg005.perseus-grc2.tb.xml",
"tlg0060.tlg001.perseus-grc3.11.tb.xml",
"tlg0540.tlg001.perseus-grc1.tb.xml",
"tlg0007.tlg004.perseus-grc1.tb.xml",
"tlg0012.tlg001.perseus-grc1.tb.xml", 
"tlg0085.tlg001.perseus-grc2.tb.xml",    
"tlg0540.tlg014.perseus-grc1.tb.xml",
"tlg0007.tlg015.perseus-grc1.tb.xml",
"tlg0012.tlg002.perseus-grc1.tb.xml",   
"tlg0085.tlg002.perseus-grc2.tb.xml",     
"tlg0540.tlg015.perseus-grc1.tb.xml",
"tlg0008.tlg001.perseus-grc1.12.tb.xml",
"tlg0013.tlg002.perseus-grc1.tb.xml",
"tlg0085.tlg003.perseus-grc2.tb.xml",    
"tlg0540.tlg023.perseus-grc1.tb.xml",
"tlg0008.tlg001.perseus-grc1.13.tb.xml",  
"tlg0016.tlg001.perseus-grc1.1.tb.xml",
"tlg0085.tlg004.perseus-grc2.tb.xml",  
"tlg0543.tlg001.perseus-grc1.tb.xml",
"tlg0011.tlg001.perseus-grc2.tb.xml",   
"tlg0020.tlg001.perseus-grc1.tb.xml",   
"tlg0085.tlg005.perseus-grc1.tb.xml",    
"tlg0548.tlg001.perseus-grc1.1.1.1-1.4.1.tb.xml",
"tlg0011.tlg002.perseus-grc2.tb.xml",
"tlg0020.tlg002.perseus-grc1.tb.xml",  
"tlg0085.tlg006.perseus-grc2.tb.xml",
"tlg0011.tlg003.perseus-grc1.tb.xml",   
"tlg0020.tlg003.perseus-grc1.tb.xml",  
"tlg0085.tlg007.perseus-grc1.tb.xml",
"tlg0011.tlg004.perseus-grc1.tb.xml",   
"tlg0059.tlg001.perseus-grc1.tb.xml",  
"tlg0096.tlg002.opp-grc2.1-53.tb.xml"]





def get_tags(path):
    r = Replacer()
    entire_treebank = path
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
                postag1 = word['postag']
                postag1 = basify(postag1)
                postag2 = word['lemma']
                postag2 = basify(postag2)
            except:
                postag = 'x--------'
            if len(form) == 0: continue
            word_tag = '/'.join([form, postag1, postag2])
            sentence_list.append(word_tag)
        sentence_str = ' '.join(sentence_list)
        sentences_list.append(sentence_str)
    treebank_training_set = '\n\n'.join(sentences_list)
    return treebank_training_set



class word:
    def __init__(self, word_string):
        self.word_string = word_string
    def get_greek(self):
        wd = self.word_string.split("/")[0]
        wd = wd.lower()
        wd = clean(wd)
        return wd
    def get_tag(self):
        return self.word_string.split("/")[1]
    def get_root(self):
        return self.word_string.split("/")[2]

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
    def add_POS(POS_to_add):
        return POS('\n\n'.join([self.POSstring,POS_to_add.POSstring]))
    def get_words_by_tag(self, tag, pos = None):
        out = []
        for _sentence in self.sentences:
            _words = _sentence.get_words()
            for _word in _words:
                try:
                    if len(_word.get_tag())>0:
                        if pos is not None:
                            if "".join([_word.get_tag()[i] for i in pos]) == tag:
                                out.append([_word.get_greek(),_word.get_tag(),_word.get_root()])
                        else:
                            if _word.get_tag() == tag:
                                out.append([_word.get_greek(),_word.get_tag(),_word.get_root()])
                except IndexError:
                    print("keep going")
        outer = out
        return outer
    def get_words(self):
        out = []
        for _sentence in self.sentences:
            _words = _sentence.get_words()
            for _word in _words:
                try:
                    if len(_word.get_tag())>0:
                        out.append([_word.get_greek(),_word.get_tag(),_word.get_root()])
                except IndexError:
                    print("keep going")
        outer = out
        return outer




text = '\n\n'.join([get_tags(root+tr) for tr in trainer])
all_str = np.asarray(POS(text).get_words())[:,0]
y = np.asarray(POS(text).get_words())[:,1]
all_string = [x for x in y]
k=pd.DataFrame(all_string)
k.columns = ['x']
o=k.groupby('x')['x'].count()
keys1 = list(o[o>10].index)
values1 = deepcopy(keys1)
keys2 = list(o[o<=10].index)
values2 = ['---------'] * len(keys2)
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
all_w = [list(word) for word in all_str]
to_tok = [" ".join(a) for a in all_w]
tok = Tokenizer()
tok.fit_on_texts(to_tok)
sequences = tok.texts_to_sequences(to_tok)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=20)


alpha = to_categorical(sequences_matrix)[1].shape[1]
max_len = 20

inputs = Input(name='inputs',shape=[max_len,alpha])
layer = LSTM(64)(inputs)
layer = Dense(256,name='FC1')(layer)
layer = Activation('relu')(layer)
layer = Dropout(0.5)(layer)
layer = Dense(len(encoded[0]),name='out_layer')(layer)
layer = Activation('softmax')(layer)
model = Model(inputs=inputs,outputs=layer)
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['categorical_accuracy'])
model.fit(to_categorical(sequences_matrix),encoded,batch_size=128,epochs=10,validation_split=0.1)



tnt_tot = tnt.TnT()
tnt_tot.train([list(zip(list(all_str),list(y)))])


model.save(path + 'pos_mini.h5')

with open(path + 'dict_letters.pkl', 'wb') as f:
    pickle.dump(dict_g, f)


with open(path + 'tokenizer.pkl', 'wb') as g:
    pickle.dump(tok, g)


with open(path + 'tnt.pkl', 'wb') as h:
    pickle.dump(tnt_tot, h)

