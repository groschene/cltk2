

trainer = [i for i in os.walk(source_txt)][0][2]
trainer.remove('daudin.xml')
#trainer = [trainer[0]]
text = ' '.join([get_tags(source_txt + tr) for tr in tqdm(trainer)])

text=text.replace('1','')
text=text.replace('�','i')
text=text.replace('2','')
text=text.replace('�','a')
text=text.replace('�','a')
text=text.replace('�','i')
text=text.replace('�','$')
text=text.replace('�','$')
text=text.replace('�','$')
text=text.replace('�','$')
text=text.replace('�','$')
text=text.replace('�','$')
text=text.replace('�','ae')
text=text.replace('�','o')


import pickle
with open('./text.pkl', 'wb') as handle:
    pickle.dump(text, handle)
