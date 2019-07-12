source_txt = '/data/q078011/cltk_data/french/text/bfm_text/BFM2019-src/'

trainer = [i for i in os.walk(source_txt)][0][2]
trainer.remove('daudin.xml')


dialects = [get_dialect(source_txt + tr) for tr in tqdm(trainer)]
list_of_dialects = [_[0] for _ in dialects]
dates = [get_date(source_txt + tr) for tr in tqdm(trainer)]
list_of_dates = [_[1][:2] for _ in dates]


le = len(trainer)

old = [i for i in range(le) if int(list_of_dates[i])<13]
modern = [i for i in range(le) if int(list_of_dates[i])>12]

to_train = [trainer[_] for _ in  old]

#trainer = [trainer[0]]
text = ' '.join([get_tags(source_txt + tr) for tr in tqdm(to_train)])




text=text.replace('1','')
text=text.replace('ÿ','i')
text=text.replace('2','')
text=text.replace('á','a')
text=text.replace('ä','a')
text=text.replace('í','i')
text=text.replace('«','$')
text=text.replace('»','$')
text=text.replace('·','$')
text=text.replace('“','$')
text=text.replace('”','$')
text=text.replace('´','$')
text=text.replace('æ','ae')
text=text.replace('ö','o')
text=text.replace(',','')
text=text.replace(':','')
text=text.replace(';','')
text=text.replace('"','')
text=text.replace('!','')

import pickle
with open('./text_old_nopnt.pkl', 'wb') as handle:
    pickle.dump(text, handle)


