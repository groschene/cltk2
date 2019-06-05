from keras.models import Model, Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, TimeDistributed, Flatten, concatenate, Reshape
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.metrics import top_k_categorical_accuracy
from attention_decoder import *
import random
from tqdm import tqdm

import pandas as pd
from copy import deepcopy
import pickle
import numpy as np
import os
import itertools

from collections import Counter
from greek_accentuation.characters import *
from nltk.tag import tnt
import nltk
from model.pos import get_tags, Pos, Sentence, Word

tokenizer = nltk.data.load('tokenizers/punkt/PY3/french.pickle')
path_to_save_model = '/home/q078011/external/dev/french_dev/model/pickled_model/dev/'
source_txt = '/data/q078011/cltk_data/french/text/bfm_text/BFM2019-src/'

valid_split = 0.1
drop_rare_pos = 20

sen_maxlen = 1000

di = {'NOMcom': 'NOM', 'VERcjg': 'VERcjg', 'PONfbl': 'PON', 'PROper': 'PROper', 'PRE': 'PRE', 'ADVgen': 'ADVgen', 'CONcoo': 'CONcoo', 'DETdef': 'DETdef', 'PONfrt': 'PON', 'ADJqua': 'ADJqua',
'CONsub': 'CONsub', 'VERppe': 'VERppe', 'VERinf': 'VERinf', 'NOMpro': 'NOM', 'PROrel': 'PROrel', 'ADVneg': 'ADVneg', 'DETpos': 'DETpos', 'PROadv': 'PROadv', 'PRE.DETdef': 'PREDETdef',
'PROdem': 'PROdem', 'PROind': 'PROind', 'DETind': 'DETind', 'DETndf': 'DETndf', 'DETdem': 'DETdem', 'PONpga': 'PON', 'PONpdr': 'PON', 'DETcar': 'DETcar', 'VERppa': 'VERppa',
'PROimp': 'PROimp', 'ADJind': 'ADJind', 'PROcar': 'PROcar', 'ABR': 'ABR', 'num': 'num', 'PROint': 'PROint', 'ADVneg.PROper': 'PROper', 'ADJcar': 'ADJcar',
'ADJpos': 'ADJpos', 'INJ': 'OTHER', 'ADVsub': 'OTHER', 'ETR': 'OTHER', 'DETrel': 'DETrel', 'OUT': 'OTHER', 'ADJord': 'ADJcar', 'PROpos': 'PROpos',
'ADVint': 'OTHER', 'ADVgen.PROper': 'ADVgenPROper', 'PROord': 'ADJcar', 'DETcom': 'DETcom', 'PROper.PROper': 'PROper', 'DETint': 'DETrel',
'PRE.DETcom': 'DETcom', 'PROrel.PROper': 'PROrelPROper', 'RED': 'PROrel', 'pon': 'PON', 'PRE.PROrel': 'OTHER', 'CONsub.PROper': 'CONsub', 
'ADVgen.PROadv': 'ADVgenPROper', 'PRE.PROper': 'PREPROper', 'PRE.DETrel': 'OTHER', 'ADVing': 'OTHER', 'PROrel.PROadv': 'OTHER',
'PROint.PROper': 'PROrel', 'ADVneg.PROadv': 'ADVneg', 'PROper.PROadv': 'PROper', 'PROcom': 'DETcom', 'PONpxx': 'PON', 'PRE.PROcom': 'OTHER'}





with open('./text.pkl', 'rb') as handle:
    text = pickle.load(handle)
    
tags = np.asarray(Pos(text).get_words())[:,1]
words = np.asarray(Pos(text).get_words())[:,0]

tk = tokenizer.tokenize(' '.join(words))
splitter = [len(x.split()) for x in tk]
indexes = list(np.cumsum(splitter))
indexes.insert(0,0)

learn_sequences = [list(words[indexes[i]:indexes[i+1]]) for i in range(len(indexes)-1)]
target_sequences = [list(tags[indexes[i]:indexes[i+1]]) for i in range(len(indexes)-1)]

tagged = [(i,t) for (i,t) in enumerate(target_sequences) if not np.max(np.asarray(t)=='None')]
to_keep = list(list(zip(*tagged))[0])
target_sequences = list(list(zip(*tagged))[1])
target_sequences = [x[:sen_maxlen] for x in target_sequences]
learn_sequences = [learn_sequences[i] for i in to_keep]
target_sequences = [[di[k] for k in t] for t in target_sequences]

flat_list_target = [item for sublist in target_sequences for item in sublist]
flat_list_learn = [item for sublist in learn_sequences for item in sublist]


to_tok = flat_list_target
tok2 = Tokenizer()
tok2.fit_on_texts(to_tok)

tar = tok2.texts_to_sequences(flat_list_target)
tar = to_categorical(tar)
all_w = [list(word) for word in flat_list_learn]
to_tok = [" ".join(a) for a in all_w]
tok = Tokenizer()
tok.fit_on_texts(to_tok)


t = tok.texts_to_sequences(to_tok)


sequences_matrix = sequence.pad_sequences(t, maxlen=15)
learn=to_categorical(sequences_matrix)


l = len(learn)
def data_generator3(l):
    r = random.choice(list(range(4,l-4)))
    X = [learn[_] for _ in [r-3,r-2,r-1,r,r+1,r+2,r+3]]
    y = tar[r]
    return X,y



def do_batch():
    x_gen1=[]
    x_gen2=[]
    x_gen3=[]
    x_gen4=[]
    x_gen5=[]
    x_gen6=[]
    x_gen7=[]
    y_gen=[]
    for _ in tqdm(range(200000)):
        u=data_generator3(l)
        x_gen1.append(u[0][0])
        x_gen2.append(u[0][1])
        x_gen3.append(u[0][2])
        x_gen4.append(u[0][3])
        x_gen5.append(u[0][4])
        x_gen6.append(u[0][5])
        x_gen7.append(u[0][6])
        x_gen = [x_gen1,x_gen2,x_gen3,x_gen4,x_gen5,x_gen6,x_gen7]
        y_gen.append(u[1])
    return (np.asarray(x_gen),np.asarray(y_gen))



def top2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def top3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

nbcar = len(tok.word_index) + 1
nbpos = len(tok2.word_index) + 1
nblstm=128

input_plus_3 = Input(shape=(15,nbcar))
input_plus_2 = Input(shape=(15,nbcar))
input_plus = Input(shape=(15,nbcar))
input_center = Input(shape=(15,nbcar))
input_minus = Input(shape=(15,nbcar))
input_minus_2 = Input(shape=(15,nbcar))
input_minus_3 = Input(shape=(15,nbcar))

lstm_plus_b = LSTM(nblstm,return_sequences = True, go_backwards=True, recurrent_dropout = 0.5)(input_plus)
lstm_plus_f = LSTM(nblstm,return_sequences = True, go_backwards=False, recurrent_dropout = 0.5)(lstm_plus_b)
attention_plus = AttentionDecoder(nblstm,nblstm, name='a1')(lstm_plus_f)
out_plus = Flatten()(attention_plus)
out_plus_1 = Dense(200, activation = 'relu')(out_plus)

lstm_center_b = LSTM(nblstm,return_sequences = True, go_backwards=True, recurrent_dropout = 0.5)(input_center)
lstm_center_f = LSTM(nblstm,return_sequences = True, go_backwards=False, recurrent_dropout = 0.5)(lstm_center_b)
attention_center = AttentionDecoder(nblstm,nblstm, name='a2')(lstm_center_f)
out_center = Flatten()(attention_center)
out_center_1 = Dense(200, activation = 'relu')(out_center)

lstm_minus_b = LSTM(nblstm,return_sequences = True, go_backwards=True, recurrent_dropout = 0.5)(input_minus)
lstm_minus_f = LSTM(nblstm,return_sequences = True, go_backwards=False, recurrent_dropout = 0.5)(lstm_minus_b)
attention_minus = AttentionDecoder(nblstm,nblstm, name='a3')(lstm_minus_f)
out_minus = Flatten()(attention_minus)
out_minus_1 = Dense(200, activation = 'relu')(out_minus)

lstm_minus_2_b = LSTM(nblstm,return_sequences = True, go_backwards=True, recurrent_dropout = 0.5)(input_minus_2)
lstm_minus_2_f = LSTM(nblstm,return_sequences = True, go_backwards=False, recurrent_dropout = 0.5)(lstm_minus_2_b)
attention_minus_2 = AttentionDecoder(nblstm,nblstm, name='a3_2')(lstm_minus_2_f)
out_minus_2 = Flatten()(attention_minus_2)
out_minus_2_1 = Dense(200, activation = 'relu')(out_minus_2)

lstm_plus_2_b = LSTM(nblstm,return_sequences = True, go_backwards=True, recurrent_dropout = 0.5)(input_plus_2)
lstm_plus_2_f = LSTM(nblstm,return_sequences = True, go_backwards=False, recurrent_dropout = 0.5)(lstm_plus_2_b)
attention_plus_2 = AttentionDecoder(nblstm,nblstm, name='a3_2_p')(lstm_plus_2_f)
out_plus_2 = Flatten()(attention_plus_2)
out_plus_2_1 = Dense(200, activation = 'relu')(out_plus_2)

lstm_minus_3_b = LSTM(nblstm,return_sequences = True, go_backwards=True, recurrent_dropout = 0.5)(input_minus_3)
lstm_minus_3_f = LSTM(nblstm,return_sequences = True, go_backwards=False, recurrent_dropout = 0.5)(lstm_minus_3_b)
attention_minus_3 = AttentionDecoder(nblstm,nblstm, name='a3_3')(lstm_minus_3_f)
out_minus_3 = Flatten()(attention_minus_3)
out_minus_3_1 = Dense(200, activation = 'relu')(out_minus_3)

lstm_plus_3_b = LSTM(nblstm,return_sequences = True, go_backwards=True, recurrent_dropout = 0.5)(input_plus_3)
lstm_plus_3_f = LSTM(nblstm,return_sequences = True, go_backwards=False, recurrent_dropout = 0.5)(lstm_plus_3_b)
attention_plus_3 = AttentionDecoder(nblstm,nblstm, name='a3_3_p')(lstm_plus_3_f)
out_plus_3 = Flatten()(attention_plus_3)
out_plus_3_1 = Dense(200, activation = 'relu')(out_plus_3)

retrieve = concatenate([out_minus_3_1, out_minus_2_1,out_minus_1,out_center_1,out_plus_1,out_plus_2_1, out_plus_3_1])
r2 = Reshape((7, 200))(retrieve)
lstm_f = LSTM(24,return_sequences = True, go_backwards=False, recurrent_dropout = 0.5)(r2)
out1 = Flatten()(lstm_f)
out_dense_1 = Dense(100, activation = 'relu')(out1)
out = Dense(nbpos,activation='softmax')(out_dense_1)

model = Model(inputs=[input_minus_3, input_minus_2,input_minus,input_center,input_plus,input_plus_2, input_plus_3],outputs=out)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc',top2, top3])


for _ in range(1000):
    d = do_batch()
    model.fit([d[0][0],d[0][1],d[0][2],d[0][3],d[0][4],d[0][5],d[0][6]],d[1],epochs=15,batch_size=64,validation_split=0.1,verbose =1)




model.save('attv2.h5')

quit()
from keras.models import load_model
model = load_model('att.h5', custom_objects={'AttentionDecoder': AttentionDecoder, 'top2':top2})


d = do_batch()
p=model.predict([d[0][0],d[0][1],d[0][2],d[0][3],d[0][4]])
predict=p.argmax(axis=1)
real=d[1].argmax(axis=1)


def rec(i):
    recall = [_ for _ in range(len(predict)) if predict[_]==i]
    verif = [real[_] for _ in recall]
    l=len([_ for _ in range(len(verif)) if verif[_]==i])/len(verif)
    return l



def prec(i):
    recall = [_ for _ in range(len(real)) if real[_]==i]
    verif = [predict[_] for _ in recall]
    l=len([_ for _ in range(len(verif)) if verif[_]==i])/len(verif)
    return l


wrong = [(predict[_],real[_]) for _ in range(len(p)) if predict[_]!=real[_]]
wrong_index = [_ for _ in range(len(p)) if predict[_]!=real[_]]

def inverse_dictionnary(mydict):
    return {v: k for k, v in mydict.items()}
    

dict = inverse_dictionnary(tok.word_index)


def to_text(_, pos):
    f = d[0][pos][_].argmax(axis=1)
    text = ''.join([dict.get(_) for _ in list(f) if _>0])
    return text

