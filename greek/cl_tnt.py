from keras.models import load_model
from model.clean import *
from nltk.tag import tnt
import pickle
from keras.preprocessing import sequence
from keras.utils import to_categorical
#path = '/home/q078011/external/dev/greek_dev/model/pickled_models/dev/'


class CltkTnt:
    def __init__(self, path):
        self.path  = path
        model = load_model(path + 'pos_mini.h5')
        with open(path + 'dict_letters.pkl', 'rb') as file:
            dict_g = pickle.load(file)
        with open(path + 'tokenizer.pkl', 'rb') as file:
            tok = pickle.load(file)
        with open(path + 'tnt.pkl', 'rb') as file:
            tnt_tot = pickle.load(file)
        alpha = len(model.get_weights()[0])
        self.alpha = alpha
        self.tok = tok
        self.tnt_tot = tnt_tot
        self.model = model
        self.dict_g = dict_g    
    def top_k_wd(self, wrd, k):
        token = self.greek_to_token(wrd)
        list_of = self.model.predict(token.reshape(1, 16, self.alpha))[0]
        keys = sorted(range(len(list_of)), key = lambda i : list_of[i])[-k:]
        inv_dict = {v: k for k, v in self.dict_g.items()}
        top_k_pred = [inv_dict.get(z) for z in keys]
        top_k_prob = [int(list_of[z] * 20) for z in keys]
        l = list(zip(top_k_pred,top_k_prob))
        to_wd = [i for i in l if i[1] > 0]
        d = [[(wrd,t[0])] * t[1] for t in to_wd]
        flat_list = [item for sublist in d for item in sublist]
        return flat_list
    def greek_to_token(self, wrd):
        input=" ".join(list(clean(basify(wrd))))
        sequences = self.tok.texts_to_sequences([input])
        sequences_matrix = sequence.pad_sequences(sequences, maxlen=16)
        out = to_categorical(sequences_matrix[0], num_classes=self.alpha).reshape(1, 16, self.alpha)
        return out
    def tag(self, st):
        to_pos =  clean(basify(st)).lower().split()
        wd_list = self.tnt_tot._wd.keys()
        to_pos_unk = [item for item in to_pos if item not in list(wd_list)]
        print("nb of unk wd "+str(len(to_pos_unk)))
        print(to_pos_unk)
        if len(to_pos_unk)>0:
            tnt_new = tnt.TnT()
            tnt_new.train([self.top_k_wd(wd, 2) for wd in to_pos_unk])
            self.tnt_tot._wd = tnt_new._wd.__add__(self.tnt_tot._wd)
        return self.tnt_tot.tag(to_pos)



