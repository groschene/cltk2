import pickle
import pandas as pd
import Levenshtein


source_data = '/home/q078011/cltk_data/french/text/graal_src/'
i =2

m1 = pickle.load(open(source_data + 'modern_'+str(i)+'.pkl', 'rb'))
o1 = pickle.load(open(source_data + 'old_'+str(i)+'.pkl', 'rb'))
ratio_u = [len(o1[i+1])/len(m1[i]) for i in range(len(o1)-1)]
lev_u=[Levenshtein.distance(o1[i+1],m1[i])/max(len(o1[i+1]),len(m1[i])) for i in range(len(o1)-1)]



m1 = pickle.load(open(source_data + 'modern_'+str(i)+'.pkl', 'rb'))
o1 = pickle.load(open(source_data + 'old_'+str(i)+'.pkl', 'rb'))
ratio_a = [len(o1[i])/len(m1[i]) for i in range(len(o1))]
lev_a =[Levenshtein.distance(o1[i],m1[i])/max(len(o1[i]),len(m1[i])) for i in range(len(o1))]

pd.Series(ratio_u).describe(percentiles = [0.05,0.95])
pd.Series(lev_u).describe(percentiles = [0.05,0.95])
pd.Series(ratio_a).describe(percentiles = [0.05,0.95])
pd.Series(lev_a).describe(percentiles = [0.05,0.95])