from lemmatizer import PosLemmatizer, AttentionLemmatizer, BackOffAttentionLemmatizer
import sys
pos_lemmatizer = PosLemmatizer('./model/pickled_models/dev/')
att_lemmatizer = AttentionLemmatizer('./model/pickled_models/dev/')
st = sys.argv[1]


boal = BackOffAttentionLemmatizer(pos_lemmatizer, att_lemmatizer)
s = boal.lemmatize(st)

print(s)
print(boal.pos_lemmatizer.CltkTnt.tag(st))




