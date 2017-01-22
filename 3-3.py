# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 14:11:15 2017

@author: luzhangqin
"""

#import nltk
#安装NLTK_DATA
#nltk.download()

from nltk.stem.wordnet import WordNetLemmatizer
#返回词元
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('gathering', 'v'))
print(lemmatizer.lemmatize('gathering', 'n'))
print(lemmatizer.lemmatize('gathering'))

#将文本拆分成句子
from nltk import word_tokenize
#从英文单词中获得符合语法的（前缀）词干的极其便利的工具
from nltk.stem import PorterStemmer
#词形还原
from nltk.stem.wordnet import WordNetLemmatizer
#词性标注
from nltk import pos_tag

wordnet_tags = ['n', 'v']

corpus = [
    'I am gathering ingredientsfor the sandwich.',
    'There were many wizards at the gathering.',
    'He ate the sandwiches',
    'Every sandwich was eaten by him'
]

stemmer = PorterStemmer()
print('Stemmed: ',
      [[stemmer.stem(token) for token in word_tokenize(document)] for document in corpus])


def lemmatize(token, tag):
    if tag[0].lower() in ['n', 'v']:
        return lemmatizer.lemmatize(token, tag[0].lower())
    return token
        
lemmatizer = WordNetLemmatizer()
tagged_corpus = [pos_tag(word_tokenize(document)) for document in corpus]
print('Lemmatized: ', [[lemmatize(token, tag) for token, tag in document] for document in tagged_corpus])
