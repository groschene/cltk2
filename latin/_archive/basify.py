# -*- coding: utf-8 -*-
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


def dict_reverse(st):
    reverse = []
    for name, age in new_dictionary.items():
        if age == st:
            reverse.append(name)
    return reverse


from cltk.stem.lemma import LemmaReplacer   
lemmatizer = LemmaReplacer('greek')
di = lemmatizer._load_replacement_patterns()
vl = list(map(basify,list(di.values())))
ky = list(map(basify,list(di.keys())))
new_dictionary = dict(zip(ky, vl))


