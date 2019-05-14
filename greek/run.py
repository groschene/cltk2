from lemmatizer import PosLemmatizer, AttentionLemmatizer
pos_lemmatizer = PosLemmatizer('./greek/model/pickled_models/dev/')
att_lemmatizer = AttentionLemmatizer('./greek/model/pickled_models/dev/')


step0 = pos_lemmatizer.dict_lemmatizer(wd)
print(step0)
if step0 == 'unk':
    step0 = pos_lemmatizer.pos_lemmatizer(wd,0)
    print(step0)

if step0 == 'unk':
    step0 =pos_lemmatizer.pos_lemmatizer(wd,1)
    print(step0)

if step0 == 'unk':
    step0 =pos_lemmatizer.levdist_lemmatizer(att_lemmatizer.sentence_to_lemma(wd)[0],0)
    print(step0)

if step0 == 'unk':
    step0 =pos_lemmatizer.levdist_lemmatizer(att_lemmatizer.sentence_to_lemma(wd)[0],1)
    print(step0)

if step0 == 'unk':
    step0 = pos_lemmatizer.pos_lemmatizer(wd,2)
    print(step0)

step0