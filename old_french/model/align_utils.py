import nltk
from xml.etree import ElementTree as ET
import copy
import Levenshtein
import numpy as np
from itertools import product


aligned = []
counter  = 0
old_out = [] 
modern_out = []

def verif_align(text_i, text_j):
    """ i = old = left """
    lenght = len(text_i)/len(text_j)
    lev = aligned_levdist(text_i, text_j)
    if lev < 0.55 and lenght < 1.2 and lenght > 0.8:
        return 'sentencesAligned'
    if lenght > 1.2:
        return 'leftTooLong'
    if lenght < 0.8:
        return 'rightTooLong'
    if lev > 0.55 and lenght < 1.2 and lenght > 0.8:
        return 'textUnaligned'

def merge_text(text1,text2):
    t = text1 + " " + text2
    return t


def change_text(i, old, modern):
    global counter
    a = verif_align(old[i], modern[i])
    if counter > 3:
        a == 'textUnaligned'
    if a == 'leftTooLong':
        modern[i] = merge_text(modern[i], modern[i + 1])
        del modern[i+1]
        change_text(i, old, modern)
        counter = counter + 1
    if a == 'rightTooLong':
        old[i] = merge_text(old[i], old[i + 1])
        del old[i+1]
        change_text(i, old, modern)
        counter = counter + 1
    if a == 'sentencesAligned':
        aligned.append([old[i],  modern[i]])
        counter = 0
        return 1
    if a == 'textUnaligned':
        return 0

def order_w(long_words_old,long_words_modern, list_old_ord, list_modern_ord):
    prod = [Levenshtein.distance(i,j) for (i,j) in product(long_words_old,long_words_modern)]
    prod = np.asarray(prod).reshape(len(long_words_old),len(long_words_modern))
    try:
        agmin = np.min(prod)
        li_old = [long_words_old[i] for i in [np.where(prod==agmin)[0][0]]]
        li_modern = [long_words_modern[i] for i in [np.where(prod==agmin)[1][0]]]
        list_old_ord.append(li_old[0])
        list_modern_ord.append(li_modern[0])
        reset_old = [long_words_old[i] for i in range(len(long_words_old)) if i not in [np.where(prod==agmin)[0][0]]]
        reset_modern = [long_words_modern[i] for i in range(len(long_words_modern)) if i not in [np.where(prod==agmin)[1][0]]]
        order_w(reset_old, reset_modern, list_old_ord, list_modern_ord)
    except ValueError:
        pass



def aligned_levdist(t1, t2):
    list_old_ord = []
    list_modern_ord = []
    w1 = t1.split()
    l = [len(i) for i in w1]
    long_word = np.where(np.asarray(l)>2)
    long_words_old = [w1[i] for i in list(long_word[0])]
    w1 = t2.split()
    l = [len(i) for i in w1]
    long_word = np.where(np.asarray(l)>2)
    long_words_modern = [w1[i] for i in list(long_word[0])]
    order_w(long_words_old,long_words_modern, list_old_ord, list_modern_ord)
    if len(long_words_modern)>len(long_words_old):
        flat_list = list_modern_ord
        xs_wd = [i for i in long_words_modern if i not in flat_list]
        flat_list.extend(xs_wd)
        ord_old = " ".join(list_old_ord)
        ord_modern = " ".join(flat_list)
        return Levenshtein.distance(ord_old,ord_modern)/max(len(ord_old),len(ord_modern))
    if len(long_words_modern) <= len(long_words_old):
        flat_list = list_old_ord
        xs_wd = [i for i in long_words_old if i not in flat_list]
        flat_list.extend(xs_wd)
        ord_modern = " ".join(list_modern_ord)
        ord_old = " ".join(flat_list)
        return Levenshtein.distance(ord_old,ord_modern)/max(len(ord_old),len(ord_modern))

