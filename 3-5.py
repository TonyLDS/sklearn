# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:02:44 2017

@author: luzhangqin
"""

from sklearn.feature_extraction.text import HashingVectorizer
corpus = ['the', 'ate', 'bacon', 'cat']
vectorizer = HashingVectorizer(n_features = 6)
print(vectorizer.transform(corpus).todense())
