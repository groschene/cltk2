from lemmatizer import PosLemmatizer, AttentionLemmatizer
pos_lemmatizer = PosLemmatizer('./model/pickled_models/dev/')
att_lemmatizer = AttentionLemmatizer('./model/pickled_models/dev/')
import sys

st = sys.argv[1]
wdl = st.split()

tags = pos_lemmatizer.CltkTnt.tag(st)
len(tags)
print(tags)
step =[]
for i in range(len(tags)):
	wd = wdl[i]
	tag = tags[i]
	print('word : ' + str(wd))
	step0 = pos_lemmatizer.dict_lemmatizer(wd)
	print('dict lemma : ' + str(step0))
	step0 = pos_lemmatizer.pos_lemmatizer(wd,tag[1],0)
	print('pos lemma 0 : ' + str(step0))
	step0 =pos_lemmatizer.pos_lemmatizer(wd,tag[1],1)
	print('pos lemma 1 : ' + str(step0))
	step0 =pos_lemmatizer.levdist_lemmatizer(att_lemmatizer.sentence_to_lemma(wd)[0],0)
	print('attention lemma 0 : ' +  str(step0))
	step0 =pos_lemmatizer.levdist_lemmatizer(att_lemmatizer.sentence_to_lemma(wd)[0],1)
	print('attention lemma 1 : ' + str(step0))
	step0 = pos_lemmatizer.pos_lemmatizer(wd,2)
	print('pos lemma 2 : '+ str(step0))
	step.append(step0)

print(step)

