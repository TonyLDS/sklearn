# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 12:40:37 2017

@author: luzhangqin
"""

import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
#from sklearn.cross_validation import cross_val_score

data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

#归一化处理
X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train)
X_test = X_scaler.transform(X_test)
y_test = y_scaler.transform(y_test)

regressor = SGDRegressor(loss = 'squared_loss')
scores = cross_val_score(regressor, X_train, y_train, cv = 5)
print('交叉验证R方: ', scores)
print('交叉验证R方均值: ', np.mean(scores))
regressor.fit_transform(X_train, y_train)
print('测试集R方： ', regressor.score(X_test, y_test))