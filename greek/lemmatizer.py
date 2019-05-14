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

from cl_tnt import CltkTnt

import collections
from difflib import *
from greek_accentuation.characters import strip_accents
from transliterate import translit
from cltk.corpus.greek.beta_to_unicode import Replacer
from model.clean import clean
from Levenshtein import distance


class PosLemmatizer:
    def __init__(self, path):
        self.CltkTnt  = CltkTnt(path)
        self.path  = path
        with open(path + 'regex.pkl', 'rb') as file:
            regex = pickle.load(file)
        self.regex = regex
        with open(path + 'dictionnary.pkl', 'rb') as file:
            dictionnary = pickle.load(file)
        self.dictionnary = dictionnary
        with open(path + 'lemma.pkl', 'rb') as file:
            lemma = pickle.load(file)
        self.lemma = lemma
    def dict_lemmatizer(self, st):
        lemmatizer = self.lemma
        try:    
            out = lemmatizer[lemmatizer[0]==st][2].values[0]
        except IndexError:
            out = 'unk'
        return out
    def pos_lemmatizer(self, kw, rk):
        tag = self.CltkTnt.tag(kw)[0][1]
        regex = self.regex
        w = self.dictionnary
        test_wd = clean(basify(kw)).lower()
        reg = [r for r in regex if r[0]==tag][0]
        to_remove_from_d = reg[1]
        pseudo_end = reg[2]
        if to_remove_from_d is not None and to_remove_from_d > 0:
            test_wd = test_wd[:-to_remove_from_d] + pseudo_end
        sh = w
        keep = np.where(np.asarray([distance(test_wd,s) for s in sh])==rk)
        if len(keep[0])>0:
            final = np.asarray(sh)[keep]
        else:
            final = 'unk'
        return final
    def dummy_lemmatizer(self, kw, tag):
        regex = self.regex
        w = self.dictionnary
        test_wd = clean(basify(kw)).lower()
        reg = [r for r in regex if r[0]==tag][0]
        to_remove_from_d = reg[1]
        pseudo_end = reg[2]
        if to_remove_from_d is not None and to_remove_from_d > 0:
            test_wd = test_wd[:-to_remove_from_d] + pseudo_end
        return test_wd
    def levdist_lemmatizer(self, kw, max=3):
        w = self.dictionnary
        no_find = True
        i = 0
        while no_find:
            test_wd = clean(basify(kw)).lower()
            keep = np.where(np.asarray([distance(test_wd,s) for s in w])==i)
            if len(keep[0]) > 0:
                no_find = False
                final = keep[0][0]
            i = i + 1
            if i > max:
                final = 'unk'
        return w[final]



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
from keras.models import load_model



class AttentionLemmatizer:
    def __init__(self, path):
        self.path  = path
        self.model = load_model(self.path+'att.h5', custom_objects={'AttentionDecoder': AttentionDecoder})
        with open(path + 'tokenizer.pkl', 'rb') as file:
            self.tok = pickle.load(file)
        self.num_classes = self.model.input_shape[2]
        self.max_len_of_word = self.model.input_shape[1]
    def attention_lemmatizer(self, target):
        sequences = self.tok.texts_to_sequences([target])
        sequences_matrix_target = sequence.pad_sequences(sequences, maxlen=self.max_len_of_word)
        pred = self.model.predict(to_categorical(sequences_matrix_target, num_classes = self.num_classes).reshape(1,self.max_len_of_word,self.num_classes))
        maxes = [np.argmax(pred[0][i]) for i in range(self.max_len_of_word)]
        res = [d.get(i) for i in maxes]
        res = "".join([i for i in res if i is not None])
        return res
    def sentence_to_lemma(self, st):
        to = clean(basify(st))
        all_w = [list(word) for word in to.split()]
        to_tok = [" ".join(a) for a in all_w]
        return [self.attention_lemmatize(target) for target in to_tok]