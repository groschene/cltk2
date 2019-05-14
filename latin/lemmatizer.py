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
    def lemmatize(self, st):
        lemmatizer = self.lemma
        return lemmatizer[lemmatizer[0]==st][2].values[0]
    def tryer(self, i, kw, tag):
        regex = self.regex
        w = self.dictionnary
        test_wd = clean(basify(kw)).lower()
        reg = [r for r in regex if r[0]==tag][0]
        to_remove_from_d = reg[1]
        pseudo_end = reg[2]
        if to_remove_from_d > 0:
            test_wd = test_wd[:-to_remove_from_d + i] + pseudo_end
        sh = w
        keep = np.where(np.asarray([distance(test_wd,s) for s in sh])==0)
        if len(keep[0])>0:
            final = np.asarray(sh)[keep]
        else:
            keep = np.where(np.asarray([distance(test_wd,s) for s in sh])==1)
            if len(keep[0])>0:
                final = np.asarray(sh)[keep]
            else:
                keep = np.where(np.asarray([distance(test_wd,s) for s in sh])==2)
                if len(keep[0])>0:
                    final = np.asarray(sh)[keep]
                else:
                    final = np.asarray(w)[self.last_chance(kw)]
        return final
    def dummy_lemma(self, kw, tag):
        regex = self.regex
        w = self.dictionnary
        test_wd = clean(basify(kw)).lower()
        reg = [r for r in regex if r[0]==tag][0]
        to_remove_from_d = reg[1]
        pseudo_end = reg[2]
        if to_remove_from_d is not None and to_remove_from_d > 0:
            test_wd = test_wd[:-to_remove_from_d] + pseudo_end
        return test_wd
    def last_chance(self, kw):
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
            if i > 3:
                final = 'unk'
        return final
    def st_to_lemma(self, st):
        ta = self.CltkTnt
        lemma = []
        tagged = ta.tag(st)
        to_pos =  clean(basify(st)).lower().split()
        len_st = len(to_pos)
        for i in range(len_st):
            try:
                le = self.lemmatize(to_pos[i])
            except IndexError:
                try:
                    le = self.tryer(0,*tagged[i])
                    if type(le) == np.str_:
                        le = le
                    elif len(le)>0:
                        le = le[0]
                    else:
                        le = 'unk'
                except IndexError:
                    le = 'unk'
            lemma.append(le)
            print(le)
        out = lemma
        return out
