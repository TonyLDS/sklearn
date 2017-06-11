# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 04:26:22 2017

@author: luzhangqin
"""

from os import walk, path
import numpy as np
import mahotas as mh
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X = []
y = []

for dir_path, dir_names, file_names in walk('orl_faces/'):
    for fn in file_names:
        if fn[-3:]  == 'pgm':
            image_filename = path.join(dir_path, fn)
            X.append(scale(mh.imread(image_filename, as_grey = True).reshape(10304).astype('float32')))
            y.append(dir_path)
            
X = np.array(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)
pca = PCA(n_components=150)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

print (' 训练集数据的原始维度是:{}'.format(X_train.shape))
print ('PCA 降维后训练集数据是:{}'.format(X_train_reduced.shape))

classifier = LogisticRegression()
accuracies = cross_val_score(classifier, X_train_reduced, y_train)
print (' 交叉验证准确率是:{} \n {}'.format(np.mean(accuracies), accuracies)
)
classifier.fit(X_train_reduced, y_train)
predictions = classifier.predict(X_test_reduced)
print (classification_report(y_test, predictions))