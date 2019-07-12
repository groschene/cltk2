import lxml.etree as ET


source_txt = '/data/q078011/cltk_data/french/text/bfm_text/BFM2019-src/'
entire_treebank = source_txt + 'oxfps.xml'
data = open(entire_treebank,'rb')
xslt_content = data.read()
xslt_root = ET.XML(xslt_content)
root = ET.XML(xslt_content)
words_list = [w for w in root.iter('{http://www.tei-c.org/ns/1.0}lb')]
##words_list = words_list[330:410]
print(len(words_list))

text = ''
for _ in words_list:
    try:
        txt = _.tail.lower()
    except AttributeError:
        txt = ''
    text = text +' '+ txt


