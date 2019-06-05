from keras.models import Model, Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, TimeDistributed, Flatten, concatenate, Reshape
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from attention_decoder import *
import random
from tqdm import tqdm

import pandas as pd
from copy import deepcopy
import pickle
import numpy as np
import os
import itertools

from greek_accentuation.characters import *
from nltk.tag import tnt
import nltk
from model.pos import get_tags, Pos, Sentence, Word

tokenizer = nltk.data.load('tokenizers/punkt/PY3/french.pickle')
path_to_save_model = '/home/q078011/external/dev/french_dev/model/pickled_model/dev/'
source_txt = '/data/q078011/cltk_data/french/text/bfm_text/BFM2019-src/'
max_len_of_word = 16
size_of_batch = 80
nb_of_epochs = 10
valid_split = 0.1
drop_rare_pos = 20
dedup = True
sen_maxlen = 1000
di = {'NOMcom': 'NOMcom', 'VERcjg': 'VERcjg', 'PONfbl': 'PONfbl', 'PROper': 'PROper', 'PRE': 'PRE', 'ADVgen': 'ADVgen', 'CONcoo': 'CONcoo', 'DETdef': 'DETdef', 'PONfrt': 'PONfrt', 'ADJqua': 'ADJqua', 'CONsub': 'CONsub', 'VERppe': 'VERppe', 'VERinf': 'VERinf', 'NOMpro': 'NOMpro', 'PROrel': 'PROrel', 'ADVneg': 'ADVneg', 'DETpos': 'DETpos', 'PROadv': 'PROadv', 'PRE.DETdef': 'PRE', 'PROdem': 'PROdem', 'PROind': 'PROind', 'DETind': 'DETind', 'DETndf': 'DETndf', 'DETdem': 'DETdem', 'PONpga': 'PONpga', 'PONpdr': 'PONpdr', 'DETcar': 'DETcar', 'VERppa': 'VERppa', 'PROimp': 'PROimp', 'ADJind': 'ADJind', 'PROcar': 'PROcar', 'ABR': 'ABR', 'num': 'num', 'PROint': 'PROint', 'ADVneg.PROper': 'PROper', 'ADJcar': 'ADJcar', 'ADJpos': 'ADJpos', 'INJ': 'OTHER', 'ADVsub': 'OTHER', 'ETR': 'OTHER', 'DETrel': 'OTHER', 'OUT': 'OTHER', 'ADJord': 'OTHER', 'PROpos': 'OTHER', 'ADVint': 'OTHER', 'ADVgen.PROper': 'OTHER', 'PROord': 'OTHER', 'DETcom': 'OTHER', 'PROper.PROper': 'OTHER', 'DETint': 'OTHER', 'PRE.DETcom': 'OTHER', 'PROrel.PROper': 'OTHER', 'RED': 'OTHER', 'pon': 'OTHER', 'PRE.PROrel': 'OTHER', 'CONsub.PROper': 'OTHER', 'ADVgen.PROadv': 'OTHER', 'PRE.PROper': 'OTHER', 'PRE.DETrel': 'OTHER', 'ADVing': 'OTHER', 'PROrel.PROadv': 'OTHER', 'PROint.PROper': 'OTHER', 'ADVneg.PROadv': 'OTHER', 'PROper.PROadv': 'OTHER', 'PROcom': 'OTHER', 'PONpxx': 'OTHER', 'PRE.PROcom': 'OTHER'}


di = {'NOMcom': 'NOM', 'VERcjg': 'VERcjg', 'PONfbl': 'PON', 'PROper': 'PROper', 'PRE': 'PRE', 'ADVgen': 'ADVgen', 'CONcoo': 'CONcoo', 'DETdef': 'DETdef', 'PONfrt': 'PON', 'ADJqua': 'ADJqua', 'CONsub': 'CONsub', 'VERppe': 'VERppe', 'VERinf': 'VERinf', 'NOMpro': 'NOM', 'PROrel': 'PROrel', 'ADVneg': 'ADVneg', 'DETpos': 'DETpos', 'PROadv': 'PROadv', 'PRE.DETdef': 'PREDETdef', 'PROdem': 'PROdem', 'PROind': 'PROind', 'DETind': 'DETind', 'DETndf': 'DETndf', 'DETdem': 'DETdem', 'PONpga': 'PON', 'PONpdr': 'PON', 'DETcar': 'DETcar', 'VERppa': 'VERppa', 'PROimp': 'PROimp', 'ADJind': 'ADJind', 'PROcar': 'PROcar', 'ABR': 'ABR', 'num': 'num', 'PROint': 'PROint', 'ADVneg.PROper': 'PROper', 'ADJcar': 'ADJcar', 'ADJpos': 'ADJpos', 'INJ': 'OTHER', 'ADVsub': 'OTHER', 'ETR': 'OTHER', 'DETrel': 'DETrel', 'OUT': 'OTHER', 'ADJord': 'ADJcar', 'PROpos': 'PROpos', 'ADVint': 'OTHER', 'ADVgen.PROper': 'ADVgenPROper', 'PROord': 'ADJcar', 'DETcom': 'DETcom', 'PROper.PROper': 'PROper', 'DETint': 'DETrel', 'PRE.DETcom': 'DETcom', 'PROrel.PROper': 'PROrelPROper', 'RED': 'PROrel', 'pon': 'PON', 'PRE.PROrel': 'OTHER', 'CONsub.PROper': 'CONsub', 'ADVgen.PROadv': 'ADVgenPROper', 'PRE.PROper': 'PREPROper', 'PRE.DETrel': 'OTHER', 'ADVing': 'OTHER', 'PROrel.PROadv': 'OTHER', 'PROint.PROper': 'PROrel', 'ADVneg.PROadv': 'ADVneg', 'PROper.PROadv': 'PROper', 'PROcom': 'DETcom', 'PONpxx': 'PON', 'PRE.PROcom': 'OTHER'}





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
"""l=list(zip(flat_list_target,flat_list_learn))



import random
def zero():
    return 0.1

random.shuffle(l,zero)

"""

from collections import Counter




to_tok = flat_list_target
tok2 = Tokenizer()
tok2.fit_on_texts(to_tok)

tar = tok2.texts_to_sequences(flat_list_target)
#padded_target = sequence.pad_sequences(tar, maxlen=sen_maxlen)
tar = to_categorical(tar)
### fit tokenizer ###
#flat_list = [item for sublist in learn_sequences for item in sublist]
all_w = [list(word) for word in flat_list_learn]
to_tok = [" ".join(a) for a in all_w]
tok = Tokenizer()
tok.fit_on_texts(to_tok)


### apply ###
#s = [[list(word) for word in l] for l in learn_sequences]
t = tok.texts_to_sequences(to_tok)

### masen = 100 ###
#t = [x[:sen_maxlen] for x in t]
sequences_matrix = sequence.pad_sequences(t, maxlen=15)
learn=to_categorical(sequences_matrix)
#padded = [np.concatenate((x, np.zeros([sen_maxlen-x.shape[0],16]))) for x in sequences_matrix]
#p=to_categorical(padded)
#shap = p.shape


"""

import random
def data_generator():
    l = len(sequences_matrix)
    r = random.choice(list(range(2,l-2)))
    ranger = list(range(r-2,r+3))
    X = [learn[x] for x in ranger]
    X = np.asarray(X)
    X = X
    y = tar[r]
    y = y
    return X,y



import random
def data_generator2():
    l = len(sequences_matrix)
    r = random.choice(list(range(2,l-2)))
    ranger = [r-2,r-1,r+1,r+2]
    X = [tar[x] for x in ranger]
    X = np.asarray(X)
    X = X
    y = tar[r]
    y = y
    return X,y


def do_batch():
    x_gen=[]
    y_gen=[]
    for _ in range(50000):
        u=data_generator2()
        x_gen.append(u[0])
        y_gen.append(u[1])
    return (np.asarray(x_gen),np.asarray(y_gen))

X = data_generator()[0]
x = np.asarray(X).shape[0]
y = np.asarray(X).shape[1]
z = np.asarray(X).shape[2]




model = Sequential()
model.add(LSTM(128,input_shape=[15, 53],return_sequences = True, go_backwards=True, recurrent_dropout = 0.3))
model.add(LSTM(128,return_sequences = True, recurrent_dropout = 0.3))
model.add(AttentionDecoder(128,128))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(37, activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(learn,tar,batch_size=256, epochs=10,validation_split=0.1, verbose =2)




from keras.layers.normalization import BatchNormalization

model = Sequential()
model.add(TimeDistributed(LSTM(128,input_shape=[x, y, z],return_sequences=True, recurrent_dropout = 0.3)))
model.add(TimeDistributed(LSTM(128,return_sequences=True,go_backwards=True, recurrent_dropout = 0.3)))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(AttentionDecoder(128,128)))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Flatten()))
model.add(TimeDistributed(Dense(100)))
model.add(LSTM(128,return_sequences = True, go_backwards=True, recurrent_dropout = 0.3))
model.add(LSTM(128,return_sequences = True, recurrent_dropout = 0.3))
model.add(AttentionDecoder(128,128))
model.add(Flatten())
model.add(Dense(37, activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])


model2 = Sequential()
model2.add(LSTM(128,input_shape=[5, 37],return_sequences = True, go_backwards=True, recurrent_dropout = 0.3))
model2.add(LSTM(128,return_sequences = True, recurrent_dropout = 0.3))
model2.add(Flatten())
model2.add(Dense(100))
model2.add(Dense(37, activation = 'softmax'))
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


pred = model.predict(learn)

def data_generator2():
    l = len(pred)
    r = random.choice(list(range(2,l-2)))
    ranger = [r-1,r,r+1,r+2]
    X = [pred[x] for x in ranger]
    X = np.asarray(X)
    X = X
    y = tar[r]
    y = y
    return X,y

"""

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


from keras.metrics import top_k_categorical_accuracy
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
#lstm_b = LSTM(nblstm,return_sequences = True, go_backwards=True, recurrent_dropout = 0.3)(r2)
lstm_f = LSTM(24,return_sequences = True, go_backwards=False, recurrent_dropout = 0.5)(r2)
#att = AttentionDecoder(5,5)(r2)
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


"""





for _ in range(100):
    model.fit(p[10000:20000], to_categorical(padded_target[10000:20000]), batch_size=size_of_batch, epochs=1,validation_split=valid_split)
    model.fit(p[20000:30000], to_categorical(padded_target[20000:30000]), batch_size=size_of_batch, epochs=1,validation_split=valid_split)
    model.fit(p[30000:40000], to_categorical(padded_target[30000:40000]), batch_size=size_of_batch, epochs=1,validation_split=valid_split)
    model.fit(p[40000:50000], to_categorical(padded_target[40000:50000]), batch_size=size_of_batch, epochs=1,validation_split=valid_split)





model = Sequential()
model.add(LSTM(256,input_shape=[16, 54],return_sequences=True,go_backwards=True, recurrent_dropout = 0.3))
model.add(LSTM(256,return_sequences=True, recurrent_dropout = 0.3))
model.add(AttentionDecoder(256,40))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(37, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


model.fit(p,to_categorical(padded_target), batch_size=size_of_batch, epochs=1,validation_split=valid_split)














np.asarray(padded)[:1000]


word_inputs = Input(shape=(None, 40,), dtype='int32', name='char_indices')
inputs = TimeDistributed(char_level_token_encoder())(word_inputs)
drop_inputs = SpatialDropout1D(parameters['dropout_rate'])(inputs)
lstm_inputs = TimestepDropout(parameters['word_dropout_rate'])(drop_inputs)
lstm_back = LSTM(y,return_sequences = True, go_backwards=True, recurrent_dropout = 0.3)(lstm_inputs)
lstm_front = LSTM(y,return_sequences = True, go_backwards=False, recurrent_dropout = 0.3)(lstm_back)
#final = AttentionDecoder(y,y)(lstm_front)
model  = Model(inputs=[word_inputs],outputs=[lstm_front])
model.compile(optimizer=Adagrad(lr=0.2, clipvalue=1),loss ='categorical_crossentropy' )


import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Input, SpatialDropout1D
from keras.layers import LSTM, CuDNNLSTM, Activation
from keras.layers import Lambda, Embedding, Conv2D, GlobalMaxPool1D
from keras.layers import add, concatenate
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, load_model
from keras.optimizers import Adagrad
from keras.constraints import MinMaxNorm
from keras.utils import to_categorical
from .custom_layers import TimestepDropout, Camouflage, Highway, SampledSoftmax


parameters = {
    'multi_processing': False,
    'n_threads': 4,
    'cuDNN': True if len(K.tensorflow_backend._get_available_gpus()) else False,
    'train_dataset': 'wikitext-2/wiki.train.tokens',
    'valid_dataset': 'wikitext-2/wiki.valid.tokens',
    'test_dataset': 'wikitext-2/wiki.test.tokens',
    'vocab': 'wikitext-2/wiki.vocab',
    'vocab_size': 28914,
    'num_sampled': 1000,
    'charset_size': 262,
    'sentence_maxlen': 100,
    'token_maxlen': 50,
    'token_encoding': 'word',
    'epochs': 10,
    'patience': 2,
    'batch_size': 1,
    'clip_value': 5,
    'cell_clip': 5,
    'proj_clip': 5,
    'lr': 0.2,
    'shuffle': True,
    'n_lstm_layers': 2,
    'n_highway_layers': 2,
    'cnn_filters': [[1, 32],
                    [2, 32],
                    [3, nblstm],
                    [4, 128],
                    [5, 256],
                    [6, 512],
                    [7, 512]
                    ],
    'lstm_units_size': 400,
    'hidden_units_size': 200,
    'char_embedding_size': 16,
    'dropout_rate': 0.1,
    'word_dropout_rate': 0.05,
    'weight_tying': True,
}


def char_level_token_encoder():
    charset_size = parameters['charset_size']
    char_embedding_size = parameters['char_embedding_size']
    token_embedding_size = parameters['hidden_units_size']
    n_highway_layers = parameters['n_highway_layers']
    filters = parameters['cnn_filters']
    token_maxlen = parameters['token_maxlen']
    # Input Layer, word characters (samples, words, character_indices)
    inputs = Input(shape=(None, 16,), dtype='int32')
    # Embed characters (samples, words, characters, character embedding)
    embeds = Embedding(input_dim=charset_size, output_dim=char_embedding_size)(inputs)
    token_embeds = []
    # Apply multi-filter 2D convolutions + 1D MaxPooling + tanh
    for (window_size, filters_size) in filters:
        convs = Conv2D(filters=filters_size, kernel_size=[window_size, char_embedding_size], strides=(1, 1),
                       padding="same")(embeds)
        convs = TimeDistributed(GlobalMaxPool1D())(convs)
        convs = Activation('tanh')(convs)
        convs = Camouflage(mask_value=0)(inputs=[convs, inputs])
        token_embeds.append(convs)
    token_embeds = concatenate(token_embeds)
    # Apply highways networks
    for i in range(n_highway_layers):
        token_embeds = TimeDistributed(Highway())(token_embeds)
        token_embeds = Camouflage(mask_value=0)(inputs=[token_embeds, inputs])
    # Project to token embedding dimensionality
    token_embeds = TimeDistributed(Dense(units=token_embedding_size, activation='linear'))(token_embeds)
    token_embeds = Camouflage(mask_value=0)(inputs=[token_embeds, inputs])
    token_encoder = Model(inputs=inputs, outputs=token_embeds, name='token_encoding')
    return token_encoder



model.save(path_to_save_model + 'pos_mini.h5')

with open(path_to_save_model + 'dict_letters.pkl', 'wb') as f:
    pickle.dump(dict_g, f)

with open(path_to_save_model + 'tokenizer.pkl', 'wb') as g:
    pickle.dump(tok, g)
"""