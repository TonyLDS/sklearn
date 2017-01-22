# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 19:45:12 2017

@author: luzhangqin
"""

from sklearn.feature_extraction import DictVectorizer

onehot_encoder = DictVectorizer()

instances = [{'city': 'New York'}, {'city': 'San Francisco'}, {'city': 'Chapel Hill'}]
#.toarray()转换为矩阵
#DictVectorizer使用fit_transform()实例化建模，并使用toarry()输出编码；
print(onehot_encoder.fit_transform(instances).toarray())