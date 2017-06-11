# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 21:00:39 2017

@author: luzhangqin
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
corpus = [
    'UNC and UNC played Duck in basketball',
    'Duck lost the basketball game',
    'I ate a sandwich',
    'Every sandwich was eaten by him'
    ]
    
#vectorizer = CountVectorizer()

#stop_words设置停用词
#使用binary设置返回二进制还是词频数；
vectorizer = CountVectorizer(binary = True, stop_words = 'english')    
    
#将稀疏矩阵转化为完整特征矩阵
#CountVectorizer、TfidfVectorizer使用fit_transform()实例化建模，
#并使用todense()输出编码，
#使用vocabulary_输出词库；
counts = vectorizer.fit_transform(corpus).todense()
print(counts)
#A mapping of terms to feature indices.
print(vectorizer.vocabulary_)
for x,y in [[0, 1], [0, 2], [1, 2]]:
    #算距离
    dist = euclidean_distances(counts[x], counts[y])
    print('文档{}与文档{}的距离{}'.format(x, y, dist))

