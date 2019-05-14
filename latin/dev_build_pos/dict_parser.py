from xml.etree import ElementTree as ET


import pickle
import collections
import pandas as pd
import numpy as np
from difflib import *

from greek_accentuation.characters import strip_accents
from transliterate import translit
from cltk.corpus.greek.beta_to_unicode import Replacer
from model.clean import clean
from Levenshtein import distance

#### Input list of words ####
r = Replacer()
tree = ET.parse('/home/q078011/cltk_data/greek/text/greek_lexica_perseus/greek_english_lexicon.xml')
root = tree.getroot()
li = [entry for entry in root.iter('entryFree')]
wrd = lambda i : clean(basify(r.beta_code([x.text for x in li[i].iter('orth')][0])))


w=[]
for i in range(len(li)):
    try:
        w.append(wrd(i))
    except AttributeError:
        pass



def reg(tag):
    simple = np.asarray(Pos(text).get_words_by_tag(tag,list(range(9))))
    simple = pd.DataFrame(simple).drop_duplicates()
    declined = list(simple[0])
    root = list(simple[2])
    root = [root[i] for i in range(len(root)) if len(declined[i])>3]
    declined = [d for d in declined if len(d)>3]
    if len(root) > 20:
        len(declined)
        rg = range(len(root))
        try:
            matches = [SequenceMatcher(None, declined[i], root[i]).find_longest_match(0,len(declined[i]),0,len(root[i])) for i in rg]
            matches_size = [m.size for m in matches]
            l_declined = [len(d) for d in declined]
            l_root = [len(r) for r in root]
            diff = [a-b for a,b in zip(l_declined,matches_size)]
            counter=collections.Counter(diff)
            counter
            to_remove_from_d = counter.most_common()[0][0]
            if to_remove_from_d>0:
                pseudo_root = [d[:-to_remove_from_d] for d in declined]
            if to_remove_from_d==0:
                pseudo_root = [d for d in declined]
            l_p_root = [len(r) for r in pseudo_root]
            diff = [a-b for a,b in zip(l_root,l_p_root)]
            counter=collections.Counter(diff)
            print(counter)
            to_remove = counter.most_common()[0][0]
            if to_remove>0:
                pseudo_end = [r[-to_remove:] for r in root]
            if to_remove==0:
                pseudo_end = ['' for r in root]
            counter=collections.Counter(pseudo_end)
            counter
            pseudo_end = counter.most_common()[0][0]
            root = [ps + pseudo_end for ps in pseudo_root]
        except IndexError:
            to_remove_from_d = None
            pseudo_end = None
    else:
        to_remove_from_d = None
        pseudo_end = None
    return (tag, to_remove_from_d, pseudo_end)


#### TODO check agments

path = '/home/q078011/external/greek_dev/model/pickled_models/dev/'
with open(path + 'dict_letters.pkl', 'rb') as file:
    dict_g = pickle.load(file)


h = [reg(tag) for tag in list(dict_g.keys())]

with open(path + 'regex.pkl', 'wb') as f:
    pickle.dump(h, f)


with open(path + 'regex.pkl', 'rb') as f:
    regex = pickle.load(f)




lemmatizer = pd.DataFrame(np.asarray(Pos(text).get_words()))
### TODO pickle text


def lemmatize(st):
    return lemmatizer[lemmatizer[0]==st][2].values[0]
def tryer(i, kw, tag):
    test_wd = clean(basify(kw)).lower()
    reg = [r for r in regex if r[0]==tag][0]
    to_remove_from_d = reg[1]
    pseudo_end = reg[2]
    test_wd = test_wd[:-to_remove_from_d + i] + pseudo_end
    sh = [x for x in w if x[:3]==test_wd[:3]]
    keep = np.where(np.asarray([distance(test_wd,s) for s in sh])==0)
    if len(keep[0])>0:
        final = np.asarray(sh)[keep]
    else:
        keep = np.where(np.asarray([distance(test_wd,s) for s in sh])==1)
        if len(keep[0])>0:
            final = np.asarray(sh)[keep]
        else:
            final = np.asarray(w)[last_chance(kw)]
    return final
def last_chance(kw):
    no_find = True
    i = 0
    while no_find:
        test_wd = clean(basify(kw)).lower()
        keep = np.where(np.asarray([distance(test_wd,s) for s in w])==i)
        if len(keep[0]) > 0:
            no_find = False
            final = keep[0][0]
        i = i + 1
    return final
def st_to_lemma(st, ta):
    lemma = []
    tagged = ta.tag(st)
    to_pos =  clean(basify(st)).lower().split()
    len_st = len(to_pos)
    for i in range(len_st):
        try:
            le = lemmatize(to_pos[i])
        except IndexError:
            try:
                le = tryer(0,*tagged[i])
                if len(le)>0:
                    le = le[0]
                else:
                    le = 'unk'
            except IndexError:
                le = 'unk'
        lemma.append(le)
        print(le)
    out = lemma
    return out



def translate(i):
    greek = r.beta_code([x.text for x in li[i].iter('orth')][0])
    english = [x.text for x in li[i].iter('tr')][0]
    print(greek,english)
    
##TODO create lemmatizer object
## add st_to_lemma to object
## add extra func