import pickle
from keras.models import load_model
from keras.metrics import top_k_categorical_accuracy
from attention_decoder import *
import pandas as pd
import numpy as np
import os
from keras.preprocessing import sequence
from keras.utils import to_categorical
from model.clean import *
import pandas as pd



def top2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

def top3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_k(mylist, k):   
    return sorted(range(len(mylist)), key = lambda i : mylist[i])[-k:]

def k_th(mylist, k):   
    return sorted(range(len(mylist)), key = lambda i : mylist[i])[-k]

def inverse_dictionnary(mydict):
    return {v: k for k, v in mydict.items()}

class OldFrenchPos:
    def __init__(self, path):
        self.path = path
        self.model = load_model(self.path + 'old.h5', custom_objects={'AttentionDecoder': AttentionDecoder, 'top2':top2, 'top3' : top3})
        with open(self.path + 'tok_txt.pkl', 'rb') as handle:
            self.tok2 = pickle.load(handle)
        with open(self.path + 'tok_pos.pkl', 'rb') as handle:
            self.tok = pickle.load(handle)
    def pred(self, r, learn):
        X = [learn[_].reshape(1,15,38) for _ in [r-3,r-2,r-1,r,r+1,r+2,r+3]]
        p = self.model.predict(X)
        return p.argmax(axis=1)[0]
    def predict(self, text):
        text=text.replace('1','')
        text=text.replace('ÿ','i')
        text=text.replace('2','')
        text=text.replace('á','a')
        text=text.replace('ä','a')
        text=text.replace('í','i')
        text=text.replace('ù','u')
        text=text.replace('«','$')
        text=text.replace('»','$')
        text=text.replace('·','$')
        text=text.replace('“','$')
        text=text.replace('”','$')
        text=text.replace('´','$')
        text=text.replace('æ','ae')
        text=text.replace('ö','o')
        text=text.replace('’','e ')
        text=text.replace("'",'e ')
        text=text.replace("\n",' ')
        text=text.replace(";",' ')
        text=text.replace(".",' ')
        text=text.replace(",",'')
        text=text.replace("-",' ')
        text=text.replace("?",' ')
        text=text.replace("  ",' ')
        text=text.lower()
        text = '. . . '+text+' . . .'
        flat_list_learn = text.split()
        all_w = [list(word) for word in flat_list_learn]
        to_tok = [" ".join(a) for a in all_w]
        self.tok2.fit_on_texts(to_tok)
        t = self.tok2.texts_to_sequences(to_tok)
        sequences_matrix = sequence.pad_sequences(t, maxlen=15)
        learn = to_categorical(sequences_matrix, len(self.tok2.word_index)+1)
        di = inverse_dictionnary(self.tok.word_index)    
        res = [self.pred(r, learn) for r in range(3,len(learn)-3)]
        pos = [di.get(_) for _ in res]
        out =  list(zip(flat_list_learn[3:-3],pos))
        return out



'''

path_model = '/home/gauthier/Documents/Python/cltk2/old_french/model/pickled_model/dev/'
path = path_model


model = load_model(path + 'old.h5', custom_objects={'AttentionDecoder': AttentionDecoder, 'top2':top2, 'top3' : top3})

with open(path_model + 'tok_txt.pkl', 'rb') as handle:
    tok2 = pickle.load(handle)

with open(path_model + 'tok_pos.pkl', 'rb') as handle:
    tok = pickle.load(handle)

'''






class VerbAttentionLemmatizer:
    def __init__(self, path):
        self.path  = path
        self.model = load_model(self.path+'att_verb.h5', custom_objects={'AttentionDecoder': AttentionDecoder})
        with open(path + 'att_verb_tok.pkl', 'rb') as file:
            self.tok = pickle.load(file)
        self.num_classes = self.model.input_shape[2]
        self.max_len_of_word = self.model.input_shape[1]
        self.d  = {v: k for k, v in dict(self.tok.word_index.items()).items()}
    def attention_lemmatizer(self, target):
        sequences = self.tok.texts_to_sequences([target])
        sequences_matrix_target = sequence.pad_sequences(sequences, maxlen=self.max_len_of_word)
        pred = self.model.predict(to_categorical(sequences_matrix_target, num_classes = self.num_classes).reshape(1,self.max_len_of_word,self.num_classes))
        maxes = [np.argmax(pred[0][i]) for i in range(self.max_len_of_word)]
        res = [self.d.get(i) for i in maxes]
        res = "".join([i for i in res if i is not None])
        return res
    def sentence_to_lemma(self, st):
        to = clean(basify(st))
        all_w = [list(word) for word in to.split()]
        to_tok = [" ".join(a) for a in all_w]
        return [self.attention_lemmatizer(target) for target in to_tok]





class NounAttentionLemmatizer:
    def __init__(self, path):
        self.path  = path
        self.model = load_model(self.path+'att_noun.h5', custom_objects={'AttentionDecoder': AttentionDecoder})
        with open(path + 'att_noun_tok.pkl', 'rb') as file:
            self.tok = pickle.load(file)
        self.num_classes = self.model.input_shape[2]
        self.max_len_of_word = self.model.input_shape[1]
        self.d  = {v: k for k, v in dict(self.tok.word_index.items()).items()}
    def attention_lemmatizer(self, target):
        sequences = self.tok.texts_to_sequences([target])
        sequences_matrix_target = sequence.pad_sequences(sequences, maxlen=self.max_len_of_word)
        X = to_categorical(sequences_matrix_target, num_classes = self.num_classes)
        X = X.reshape(1,self.max_len_of_word,self.num_classes)
        pred = self.model.predict(X)
        maxes = [np.argmax(pred[0][i]) for i in range(self.max_len_of_word)]
        res = [self.d.get(i) for i in maxes]
        res = "".join([i for i in res if i is not None])
        return res
    def sentence_to_lemma(self, st):
        to = clean(basify(st))
        all_w = [list(word) for word in to.split()]
        to_tok = [" ".join(a) for a in all_w]
        return [self.attention_lemmatizer(target) for target in to_tok]




class OldFrenchAttentionLemmatizer:
    def __init__(self, path_model, path_frol):
        self.path_model  = path_model
        self.path_frol  = path_frol
        self.v = VerbAttentionLemmatizer(self.path_model)
        self.n = NounAttentionLemmatizer(self.path_model)
        frolex = pd.read_csv(self.path_frol+'frolex.tsv', sep='\t')
        frolex = frolex[frolex['lemma']!='<nolem>']
        frolex = frolex[frolex['lemma'].notnull()]
        frolex.loc[~frolex['msd_bfm'].isnull(),'msd_cattex_conv']=frolex['msd_bfm']
        frolex = frolex.assign(cattex_short = pd.Series(frolex['msd_cattex_conv'].apply(lambda x : x[:3].lower())).values)
        frolex = frolex.assign(cattex_low = pd.Series(frolex['msd_cattex_conv'].apply(lambda x : x.lower().replace('.',''))).values)
        self.frolex = frolex
    def lemmatize(self, w, pos):
        frolex = self.frolex
        v = self.v
        n = self.n
        lemma = ''
        size = frolex[(frolex['form']==w)]['lemma'].drop_duplicates().size
        if size == 0:
            if pos[:3] == 'ver':
                lemma = v.sentence_to_lemma(w)
            if pos[:3] != 'ver':
                lemma = n.sentence_to_lemma(w)
        if size == 1:
            lemma = frolex[(frolex['form']==w)]['lemma'].drop_duplicates().values[0]
        if size > 1:
            size_cross = frolex[(frolex['form']==w) & (frolex['cattex_short'] == pos[:3])]['lemma'].drop_duplicates().size
            if size_cross == 0:
                lemma = frolex[(frolex['form']==w)]['lemma'].drop_duplicates().values[0]
            if size_cross > 0:
                lemma = frolex[(frolex['form']==w) & (frolex['cattex_short'] == pos[:3])]['lemma'].drop_duplicates().values[0]
        return lemma




 
#########################todo FROLEX
'''
lemma -> drop <nolem> NaN
msd_cattex_conv == msd_bfm if not NaN

frolex[(frolex['form']=='voluntet')] == 1 -> OK if lemma
if no lemma -> dico
if no lemma -> IdentityLemma

else > 1
frolex[(frolex['form']=='voluntet') & (frolex['msd_cattex_conv']=='pos')] == 1 -> OK
if no lemma -> other lemma
if no lemma -> dico
if no lemma -> form = AttentionLemmatizer
if no lemma -> IdentityLemma

from attention_decoder import *
from keras.models import Sequential
from keras.layers import LSTM
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Flatten
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer



import pandas as pd

path_frol = '/home/gauthier/Documents/Python/mftk/'
frolex = pd.read_csv(path_frol+'frolex.tsv', sep='\t')
frolex = frolex[frolex['lemma']!='<nolem>']
frolex = frolex[frolex['lemma'].notnull()]
frolex.loc[~frolex['msd_bfm'].isnull(),'msd_cattex_conv']=frolex['msd_bfm']
frolex = frolex.assign(cattex_short = pd.Series(frolex['msd_cattex_conv'].apply(lambda x : x[:3].lower())).values)
frolex = frolex.assign(cattex_low = pd.Series(frolex['msd_cattex_conv'].apply(lambda x : x.lower().replace('.',''))).values)




verbs = frolex[frolex['cattex_short']!='nom']

verbs = verbs[verbs['form']!='nan']
verbs = verbs[verbs['form']!=np.nan]
verbs = verbs[verbs['form'].notnull()]


form = verbs.form.values
lemmas = verbs.lemma.values


all_w = [list(word) for word in list(form)]
to_tok = [" ".join(a) for a in all_w]
max_len_of_word = 16
tok = Tokenizer()
tok.fit_on_texts(to_tok)
sequences = tok.texts_to_sequences(to_tok)
sequences_matrix_in = sequence.pad_sequences(sequences, maxlen=max_len_of_word)



all_w = [list(word) for word in list(lemmas)]
to_tok = [" ".join(a) for a in all_w]

sequences = tok.texts_to_sequences(to_tok)
sequences_matrix_target = sequence.pad_sequences(sequences, maxlen=max_len_of_word)



with open(path_model + 'att_noun_tok.pkl', 'wb') as h:
    pickle.dump(tok, h)

# define model

model = Sequential()
model.add(LSTM(150, input_shape=(16, 60), return_sequences=True))
model.add(LSTM(150, go_backwards=True, return_sequences=True))
model.add(AttentionDecoder(150, 60))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(to_categorical(sequences_matrix_in,num_classes=60), to_categorical(sequences_matrix_target,num_classes=60),batch_size=64, validation_split=0.1, epochs=14)

w = 'eures'
w = 'finissez'
sequences = tok.texts_to_sequences([' '.join(list(w))])
sequences_matrix_target = sequence.pad_sequences(sequences, maxlen=16)
pred = model.predict(to_categorical(sequences_matrix_target, num_classes = 49).reshape(1,16,49))
maxes = [np.argmax(pred[0][i]) for i in range(16)]
res = [d.get(i) for i in maxes]
res = "".join([i for i in res if i is not None])
res


model.save(path_model + 'att_noun.h5')
'''

############################


