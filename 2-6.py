# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 19:15:49 2016

@author: luzhangqin
"""
import matplotlib.pylab as plt
from matplotlib.font_manager import FontProperties

import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures

#显示中文
font = FontProperties(fname = r'/usr/share/fonts/truetype/arphic/ukai.ttc', size = 10)

def runplt():
    #创建一个用来显示图形输出的一个窗口对象
    plt.figure()
    plt.title('批萨价格与直径数据', fontproperties = font)
    plt.xlabel('直径（英寸）', fontproperties = font)
    plt.ylabel('价格（美元）', fontproperties = font)
    #[xmin, xmax, ymin, ymax]
    plt.axis([0, 25, 0, 25])
    #网格
    plt.grid(True)
    return plt

X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]

X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]

#一元线性回归 regressor
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#linspace 创建等差数列
#第一个参数表示起始点、第二个参数表示终止点，第三个参数表示数列的个数。
xx = np.linspace(0, 26, 100)
#shape 它的功能是读取矩阵的长度
#reshape更改数组的形状
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt = runplt()
plt.plot(X_train, y_train, 'k.')
plt.plot(xx, yy)

#二次回归 quadratic
#degree 多项式的阶数
#interaction_only 如果值为true(默认是false),则会产生相互影响的特征集。
#include_bias 是否包含偏差列 True (default)
quadratic_featurizer = PolynomialFeatures(degree= 2)
#fit 训练算法
#transform 数据转换
#fit_transform 合并 fit和transform
#y = ax^2 + bx + c
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
X_test_quadratic = quadratic_featurizer.transform(X_test)
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), 'r-')

#三次回归 cubic
cubic_featurizer = PolynomialFeatures(degree= 3)
X_train_cubic = cubic_featurizer.fit_transform(X_train)
X_test_cubic = cubic_featurizer.transform(X_test)
regressor_cubic = LinearRegression()
regressor_cubic.fit(X_train_cubic, y_train)
xx_cubic = cubic_featurizer.transform(xx.reshape(xx.shape[0], 1))
plt.plot(xx, regressor_cubic.predict(xx_cubic), 'y-')

#七次回归 seventh
seventh_featurizer = PolynomialFeatures(degree= 7)
X_train_seventh = seventh_featurizer.fit_transform(X_train)
X_test_seventh = seventh_featurizer.transform(X_test)
regressor_seventh = LinearRegression()
regressor_seventh.fit(X_train_seventh, y_train)
xx_seventh = seventh_featurizer.transform(xx.reshape(xx.shape[0], 1))
plt.plot(xx, regressor_seventh.predict(xx_seventh), 'g-')


plt.show()

print(X_train)
print(X_train_quadratic)
print(X_test)
print(X_test_quadratic)
print('一元线性回归 r-squared: ', regressor.score(X_test, y_test))
print('二次回归 r-squared: ',regressor_quadratic.score(X_test_quadratic, y_test))
print('三次回归 r-squared: ',regressor_cubic.score(X_test_cubic, y_test))
print('七次回归 r-squared: ',regressor_seventh.score(X_test_seventh, y_test))

