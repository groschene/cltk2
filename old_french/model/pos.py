from greek_accentuation.characters import *
from model.clean import *
from cltk.corpus.greek.beta_to_unicode import Replacer
from lxml import etree

class Word:
    def __init__(self, word_string):
        self.word_string = word_string
    def get_greek(self):
        wd = self.word_string.split("/")[0]
        wd = wd.lower()
        wd = clean(basify(wd))
        return wd
    def get_tag(self):
        return self.word_string.split("/")[1]
    def get_root(self):
        wd = self.word_string.split("/")[2]
        wd = wd.lower()
        wd = clean(basify(wd))
        return wd


class Sentence:
    def __init__(self, sentence_string):
        self.sentence_string = sentence_string
    def get_words(self):
        words = [Word(s) for s in self.sentence_string.split(" ")]
        return words


class Pos:
    def __init__(self, pos_string):
        self.pos_string = pos_string
        self.sentences = self.get_sentences()
    def get_sentences(self):
        sentences = [Sentence(s) for s in self.pos_string.split("\n\n")]
        return sentences
    def add_pos(self, pos_to_add):
        return Pos('\n\n'.join([self.pos_string, pos_to_add.pos_string]))
    def get_words_by_tag(self, tag, pos=None):
        out = []
        for _sentence in self.sentences:
            _words = _sentence.get_words()
            for _word in _words:
                try:
                    if len(_word.get_tag()) > 0:
                        if pos is not None:
                            if "".join([_word.get_tag()[i] for i in pos]) == tag:
                                out.append([_word.get_greek(), _word.get_tag(), _word.get_root()])
                        else:
                            if _word.get_tag() == tag:
                                out.append([_word.get_greek(), _word.get_tag(), _word.get_root()])
                except IndexError:
                    print("keep going")
        outer = out
        return outer
    def get_words(self):
        out = []
        for _sentence in self.sentences:
            _words = _sentence.get_words()
            for _word in _words:
                try:
                    if len(_word.get_tag()) > 0:
                        out.append([_word.get_greek(), _word.get_tag(), _word.get_root()])
                except IndexError:
                    print("keep going")
        outer = out
        return outer

def get_tags(path):
    entire_treebank = path
    data = open(entire_treebank,'rb')
    xslt_content = data.read()
    xslt_root = etree.XML(xslt_content)
    root = etree.XML(xslt_content)
    words_list = [w for w in root.iter('{http://www.tei-c.org/ns/1.0}w')]
    print(len(words_list))
    word_tag = []
    for x in words_list:
        try:
            lemmas = x.attrib.get('lemma')
        except IndexError:
            lemmas = 'no_lemma'
        try:
            ana = x.attrib.get('type')
        except IndexError:
            ana = 'no_pos'
        word = x.text
        wt = '/'.join([str(word), str(ana), str(lemmas)])
        word_tag.append(wt)
    sentence_str = ' '.join(word_tag)
    out = sentence_str
    return out



def get_date(path):
    entire_treebank = path
    data = open(entire_treebank,'rb')
    xslt_content = data.read()
    xslt_root = etree.XML(xslt_content)
    root = etree.XML(xslt_content)
    words_list = [w for w in root.iter('{http://www.tei-c.org/ns/1.0}date')]
    dates = []
    for x in words_list:
        try:
            date = x.attrib.get('when')
        except IndexError:
            date = 'no_date'
        dates.append(date)
    return dates




def get_dialect(path):
    entire_treebank = path
    data = open(entire_treebank,'rb')
    xslt_content = data.read()
    xslt_root = etree.XML(xslt_content)
    root = etree.XML(xslt_content)
    words_list = [w for w in root.iter('{http://www.tei-c.org/ns/1.0}region')]
    dates = []
    for x in words_list:
        try:
            reg = x.text
        except IndexError:
            reg = 'standard'
        dates.append(reg)
    return dates




### Process ana