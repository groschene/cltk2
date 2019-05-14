import nltk
from xml.etree import ElementTree as ET
import copy
import Levenshtein
import numpy as np
from itertools import product
from model.align_utils import *

source_data = '/home/q078011/cltk_data/french/text/graal_src/'
source_txt = 'qgraal_cm-bfm.xml'

### parse XML
tree = ET.parse(source_data + source_txt)
root = tree.getroot()
full_pos = [entry.attrib['pos'] for entry in root.iter('{http://www.tei-c.org/ns/1.0}w')]
full_txt = [entry.text for entry in root.iter('{http://www.tei-c.org/ns/1.0}w')]
full = [entry for entry in root.iter('{http://www.tei-c.org/ns/1.0}w')]
joiner = lambda k : "".join([i for i in full[k].itertext()])
old_raw = " ".join([joiner(k) for k in range(len(full))])

### get modern text
with open(source_data + 'queste_modern.txt') as f:
    modern_raw = f.read()

### sample

def full_raw_text(old, modern):
    modern = modern.replace("\n"," ")
    modern = modern.replace("<160a>"," ")
    modern = modern.replace("(français moderne) ","")
    modern = modern.replace(" sur 168 11/11/2012 18:14","")
    modern = modern.replace("Quête du saint Graal ","") 
    old = old.replace(".i.","")
    old = old.replace(".ii.","")
    old = old.replace(".iii.","")
    old = old.replace(".iiii.","")
    old = old.replace(".v.","")
    old = old.replace(".vi.","")
    old = old.replace(".vii.","")
    old = old.replace(".viii.","")
    old = old.replace(".ix.","")
    old = old.replace(".x.","")
    old = old.replace(".xi.","")
    old = old.replace(".xii.","")
    old = old.replace(".xiii.","")
    old = old.replace(".xiv.","")
    old = old.replace(".xv.","")
    old = old.replace(".xvi.","")
    return old, modern



def tknz_str(old, modern):
    tokenizer = nltk.data.load('tokenizers/punkt/PY3/french.pickle')
    old = tokenizer.tokenize(old)
    modern = tokenizer.tokenize(modern)
    return old, modern




aligned = []
counter  = 0
old_out = [] 
modern_out = []

old_fixed, modern_fixed = full_raw_text(old_raw, modern_raw)
old, modern = tknz_str(old_fixed, modern_fixed)

### do LOOP

for _ in range(30):
    stop = 0
    i = 0
    while stop == 0:
        list_old_ord = []
        list_modern_ord = []
        if verif_align(old[i], modern[i]) != 'sentencesAligned':
             a = change_text(i, old, modern)
             if a == 0:
                 stop = 1
        i = i + 1
    f = i
    print(f)    
    old_out.append(old[:f-1])
    modern_out.append(modern[:f-1])
    j = f + 1
    st1 = old_fixed.find(old[j])
    st2 = modern_fixed.find(modern[j])
    a=[Levenshtein.distance(old[j], modern_fixed[i:i+int(len(old[j])*1.1)]) for i in range(st2-10000,st2+10000)]
    i = st2-10000+np.argmin(a)
    new_modern_start = modern_fixed.find(modern_fixed[i:i+int(len(old[j])*1.1)])
    new_old_start = old_fixed.find(old[j])
    old, modern = tknz_str(old_fixed[new_old_start:], modern_fixed[new_modern_start:])
    if len(modern[0]) < 10:
        del modern[0]  
    if len(old[0]) < 10:
        del old[0]

modern_flat_list = [item for sublist in modern_out for item in sublist]
old_flat_list = [item for sublist in old_out for item in sublist]




