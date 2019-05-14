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





#with open('greek_training_set_2.pos', 'w') as f:
#f.write(treebank_training_set)

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


##TODO create POS object
## create word object

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



text = '\n\n'.join([get_tags(root+tr) for tr in trainer])


all_str = np.asarray(POS(text).get_words())[:,0]
y = np.asarray(POS(text).get_words())[:,1]




all_string = [x for x in y]

preprocess = True

if preprocess == True : 
    k=pd.DataFrame(all_string)
    k.columns = ['x']
    o=k.groupby('x')['x'].count()
    keys1 = list(o[o>100].index)
    values1 = deepcopy(keys1)
    keys2 = list(o[o<=100].index)
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




def french_to_token(wrd):
    input=" ".join(list(g_translit(wrd)))
    sequences = tok.texts_to_sequences([input])
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=20)
    out = to_categorical(sequences_matrix[0],num_classes=alpha).reshape(1,20,alpha)
    return out



def greek_to_token(wrd):
    input=" ".join(list(clean(basify(wrd))))
    sequences = tok.texts_to_sequences([input])
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=20)
    out = to_categorical(sequences_matrix[0],num_classes=alpha).reshape(1,20,alpha)
    return out



def predictor(token):
    arg = model.predict(token.reshape(1,20,alpha)).argmax()
    print(list(dict_g.keys())[list(dict_g.values()).index(arg)])


def top_k(token,k):
    list_of = model.predict(token.reshape(1,20,alpha)) ### return top k pred
    keys = [sorted(range(len(list_of[j])), key=lambda i: list_of[j][i])[-k:] for j in range(len(list_of))]
    top_k_pred = [list(dict_g.keys())[list(dict_g.values()).index(z)] for z in keys[0]]
    return top_k_pred

def remove_z(list):
    i=0
    length = len(list)
    while(i<length):
        if(list[i]==0):
          	list.remove (list[i])
          	length = length -1  
          	continue
        i = i+1
    return list



def decode(token):
    from_dict = [token[0][i].argmax() for i in range(20)]
    from_dict = remove_z(from_dict)
    return "".join([list(tok.word_index.keys())[list(tok.word_index.values()).index(z)] for z in from_dict])




def top_k_wd(wrd,k):
    token = greek_to_token(wrd)
    list_of = model.predict(token.reshape(1,20,alpha)) ### return top k pred
    keys = [sorted(range(len(list_of[j])), key=lambda i: list_of[j][i])[-k:] for j in range(len(list_of))]
    top_k_pred = [list(dict_g.keys())[list(dict_g.values()).index(z)] for z in keys[0]]
    top_k_prob = [int(list_of[0][z]*10) for z in keys[0]]
    l = list(zip(top_k_pred,top_k_prob))
    to_wd = [i for i in l if i[1]>0]
    d = [[(wrd,t[0])] * t[1] for t in to_wd]
    flat_list = [item for sublist in d for item in sublist]
    return flat_list



alpha = to_categorical(sequences_matrix)[1].shape[1]

from keras.metrics import top_k_categorical_accuracy
k3 = lambda x, y: top_k_categorical_accuracy(x, y, k=3)
k2 = lambda x, y: top_k_categorical_accuracy(x, y, k=2)
k5 = lambda x, y: top_k_categorical_accuracy(x, y, k=5)
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
model.fit(to_categorical(sequences_matrix),encoded,batch_size=128,epochs=5,validation_split=0.1)


model.save('/home/q078011/external/greek_dev/pos_mini.h5')









st = '???t? d? ???? ?????e?, ??a p?????? t? ????? ?p? t?? ?????? d?? t?? p??f?t??, ?????t??'
to_pos =  clean(basify(st))


tnt_tot = tnt.TnT()
tnt_tot.train([list(zip(list(all_str),list(y)))])


import pickle

with open('/home/q078011/external/greek_dev/dict_letters.pkl', 'wb') as f:
    pickle.dump(dict_g, f)


with open('/home/q078011/external/greek_dev/tokenizer.pkl', 'wb') as g:
    pickle.dump(tok, g)


with open('/home/q078011/external/greek_dev/tnt.pkl', 'wb') as h:
    pickle.dump(tnt_tot, h)



tnt_new = tnt.TnT()
tnt_new.train([top_k_wd(wd,15) for wd in to_pos.split()])
tnt_tot._wd = tnt_new._wd.__add__(tnt_tot._wd)
tnt_tot.tag(to_pos.split())

##########################
###     Simplify        ###
##########################


import pandas as pd
simple = np.asarray(POS(text).get_words_by_tag('v3piia---',list(range(9))))
simple = pd.DataFrame(simple).drop_duplicates()

def common(x,y):
    rev_x=x[::-1]
    rev_y=y[::-1]
    root = []
    i = 0
    end = 0
    while (end == 0) & (i<len(y)) & (i<len(x)):
        if rev_x[i] == rev_y[i]:
            root.append(rev_x[i])
            i = i + 1
        else:
            end = 1
    if len(root)>0:
        r="".join(root)[::-1]
    else:
        r=""
    return r



ao = np.asarray(POS(text).get_words_by_tag('v1sa',[0,1,2,3]))[:1000]
ao = ao[:,0]
a=[common(i,j) for i,j in product(ao,ao)]
from itertools import groupby
import collections

counter=collections.Counter(a)







### TODO match pseudo root to dict



##########################
###     SEQ2SEQ        ###
##########################




## extract trainer
text = '\n\n'.join([get_tags(root+tr) for tr in trainer])



## get strings
all_str = np.asarray(POS(text).get_words())[:,0]
y = np.asarray(POS(text).get_words())[:,2]

tag = np.asarray(POS(text).get_words())[:,1]
keep = [t[0]=='v' for t in tag]

all_str = all_str[keep]
y = y[keep]

## get all letters
all_w = [list(word) for word in all_str] ## one list per word
flat_list = [item for sublist in all_w for item in sublist]
woduplicates = list(set(flat_list))


tokenizer = Tokenizer()
to_tok = [" ".join(a) for a in all_w]
max_len=20
tokenizer.fit_on_texts(to_tok)
sequences = tokenizer.texts_to_sequences(to_tok)
sequences_matrix_x = sequence.pad_sequences(sequences, maxlen=max_len)


alpha = to_categorical(sequences_matrix_x)[1].shape[1]


sequences_matrix_in = to_categorical(sequences_matrix_x, alpha)


all_w_y = [list(word) for word in y]
to_tok_y = [" ".join(a) for a in all_w_y]
sequences_y = tokenizer.texts_to_sequences(to_tok_y)
sequences_matrix_y = sequence.pad_sequences(sequences_y, maxlen=max_len)
sequences_matrix_tar = to_categorical(sequences_matrix_y, alpha)


sequences_matrix_tar_in = np.zeros(sequences_matrix_tar.shape)

for i in range(19):
    sequences_matrix_tar_in[:,i+1,:] = sequences_matrix_tar[:,i,:]


## learn

encoder_inputs = Input(shape=[max_len,alpha])
encoder = LSTM(64, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]
decoder_inputs = Input(shape=[None,alpha])
decoder_lstm = LSTM(64, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,initial_state=encoder_states)
decoder_dense = Dense(alpha, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([sequences_matrix_in, sequences_matrix_tar_in], sequences_matrix_tar, batch_size=128, epochs=10, validation_split=0.2)

## inference
encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape=(64,))
decoder_state_input_c = Input(shape=(64,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)



def rooter(fr_tok):
    input_seq = french_to_token(fr_tok)
    tokens = []
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, alpha))
    target_seq[0, 0, 0] = 1.
    stop_condition = False
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        tokens.append(sampled_token_index)
        if (len(tokens) > 19):
            stop_condition = True
        target_seq = np.zeros((1, 1, alpha))
        target_seq[0, 0, sampled_token_index] = 1.
        states_value = [h, c]
    pred = "".join([list(tok.word_index.keys())[list(tok.word_index.values()).index(z)] for z in remove_z(tokens)])
    return pred






## extract trainer
text = '\n\n'.join([get_tags(root+tr) for tr in trainer])



## get strings
all_str = np.asarray(POS(text).get_words())[:,0]
y = np.asarray(POS(text).get_words())[:,2]

tag = np.asarray(POS(text).get_words())[:,1]
keep = [t[0]=='v' for t in tag]
all_str = all_str[keep]
y = y[keep]

## get all letters
all_w = [list(word) for word in all_str] ## one list per word
flat_list = [item for sublist in all_w for item in sublist]
woduplicates = list(set(flat_list))


tokenizer = Tokenizer()
to_tok = [" ".join(a) for a in all_w]
max_len=20
tokenizer.fit_on_texts(to_tok)
sequences = tokenizer.texts_to_sequences(to_tok)
sequences_matrix_x = sequence.pad_sequences(sequences, maxlen=max_len)


alpha = to_categorical(sequences_matrix_x)[1].shape[1]


sequences_matrix_in = to_categorical(sequences_matrix_x, alpha)


all_w_y = [list(word) for word in y]
to_tok_y = [" ".join(a) for a in all_w_y]
sequences_y = tokenizer.texts_to_sequences(to_tok_y)
sequences_matrix_y = sequence.pad_sequences(sequences_y, maxlen=max_len)
sequences_matrix_tar = to_categorical(sequences_matrix_y, alpha)


sequences_matrix_in = sequences_matrix_in.reshape(557320, 20, 41, 1)
sequences_matrix_tar = sequences_matrix_tar.reshape(557320, 20, 41, 1)
from keras.layers import MaxPooling2D, Conv2D, UpSampling2D, Dropout, Input
input_txt = Input(shape = (20, 41, 1))
conv1 = Conv2D(8, (5, 41), activation='relu', padding='same')(input_txt)

decoded = Conv2D(1, (5, 41), activation='sigmoid', padding='same')(conv1)

autoencoder = Model(input_txt, decoded)
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
autoencoder.summary()
autoencoder.fit(sequences_matrix_in,sequences_matrix_tar,batch_size=128,epochs=5,validation_split=0.5)











### TODO NN for lemmatizing -> Direct forecast
########
## TODO simplifier regex based + levdist on dict
## TODO only add unk in POS
## TODO Remove dummy lemmatizer in backoff chained lemmatizer





