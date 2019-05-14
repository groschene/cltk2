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
    r = Replacer()
    entire_treebank = path
    with open(entire_treebank, 'r') as f:
        xml_string = f.read()
    root = etree.fromstring(xml_string)
    body = root.findall('body')[0]
    sentences = body.findall('sentence')
    sentences_list = []
    for sentence in sentences:
        words_list = sentence.findall('word')
        sentence_list = []
        for x in words_list:
            word = x.attrib
            form = word['form'].upper()
            form = r.beta_code(form)
            try:
                if form[-1] == 's':
                    form = form[:-1] + '?'
            except IndexError:
                pass
            form = form.lower()
            form = clean(basify(form))
            form_list = [char for char in form if char not in [' ', "'", '?', '’', '[', ']']]
            form = ''.join(form_list)
            try:
                postag1 = word['postag']
                postag1 = postag1
                postag2 = word['lemma']
                postag2 = clean(basify(postag2))
            except:
                postag = 'x--------'
            if len(form) == 0: continue
            word_tag = '/'.join([form, postag1, postag2])
            sentence_list.append(word_tag)
        sentence_str = ' '.join(sentence_list)
        sentences_list.append(sentence_str)
    treebank_training_set = '\n\n'.join(sentences_list)
    return treebank_training_set

