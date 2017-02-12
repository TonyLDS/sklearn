# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 22:23:31 2017

@author: luzhangqin
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
#1.8删除并且移动到model_selection模块中
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split, cross_val_score
#为了二值化
from sklearn import preprocessing

from sklearn.metrics import roc_curve, auc
import matplotlib.pylab as plt


df = pd.read_csv('smsspamcollection/SMSSpamCollection', sep = '\t', header = None)
print(df.head())
print('含spam短信数量:', df[df[0] == 'spam'][0].count())
print('含ham短信数量:', df[df[0] == 'ham'][0].count())



X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1], df[0])

#二值化
lb = preprocessing.LabelBinarizer()
y_train = np.array([number[0] for number in lb.fit_transform(y_train)])
y_test = np.array([number[0] for number in lb.fit_transform(y_test)])

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

classifier = LogisticRegression()

#准确率
scores = cross_val_score(classifier, X_train, y_train, cv=5)
print('准确率: ', np.mean(scores), scores)

#精确率和召回率
precisions = cross_val_score(classifier, X_train, y_train, cv=5, scoring  = 'precision')
print('精确率: ', np.mean(precisions), precisions)

recalls = cross_val_score(classifier, X_train, y_train, cv=5, scoring  = 'recall')
print('召回率: ', np.mean(recalls), recalls)

#综合评价指标
#有时也会用F0.5和F2，表示精确率权重大于召回率，或召回率权重大于精确率
f1s = cross_val_score(classifier, X_train, y_train, cv=5, scoring  = 'f1')
print('计算机综合指标： ',np.mean(f1s), f1s)

#ROC and AUC
classifier.fit(X_train, y_train)
predictions = classifier.predict_proba(X_test)

FP_rate, recall, thresholds = roc_curve(y_test, predictions[:, 1])
roc_auc = auc(FP_rate, recall)

plt.title('Receiver Operating Characteristic')
plt.plot(FP_rate, recall, 'b', label = "AUC = %.2f" %roc_auc)
#在轴上放置一个图例。
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()