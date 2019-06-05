

trainer = [i for i in os.walk(source_txt)][0][2]
trainer.remove('daudin.xml')
#trainer = [trainer[0]]
text = ' '.join([get_tags(source_txt + tr) for tr in tqdm(trainer)])

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


import pickle
with open('./text.pkl', 'wb') as handle:
    pickle.dump(text, handle)
