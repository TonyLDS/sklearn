# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 23:25:23 2016

@author: luzhangqin
"""

import pandas as pd
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
#分训练集和测试集
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split

df = pd.read_csv('winequality-red.csv', sep =';')
#变量名并排除最后一个
X = df[list(df.columns)[:-1]]
y = df['quality']
#分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_predictions = regressor.predict(X_test)
print('R-squared: ', regressor.score(X_test, y_test))

###############################################################
#交叉验证
from sklearn.model_selection import cross_val_score
#from sklearn.cross_validation import cross_val_score
regressor = LinearRegression()
#交叉验证
scores = cross_val_score(regressor, X, y, cv = 5)
#均值
print(scores.mean())
print(scores)

###################################################################
plt.scatter(y_test, y_predictions)
plt.xlabel('y_test')
plt.xlabel('y_predictions')
plt.title('y_test and y_predictions')
plt.show()
