# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 20:15:54 2017

@author: luzhangqin
"""
#计数器
from sklearn.feature_extraction.text import CountVectorizer
corpus = ['The dog ate a sandwich, the wizard transfigured a sandwich, and I ate sandwich']
vectorizer = CountVectorizer(stop_words = 'english')
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)

from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'The dog ate a sandwich and I ate a sandwich,'
    'The wizard transfigured a sandwich.'
    ]
vectorizer = TfidfVectorizer(stop_words = 'english')
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)