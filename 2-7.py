# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 22:35:35 2016

@author: luzhangqin
"""

import pandas as pd
import matplotlib.pylab as plt

df = pd.read_csv('winequality-red.csv', sep = ';')
#使用head查看前几行数据(默认是前5行)
print(df.head())
#会统计出各列的：计数，平均数，标准差，最小值，最大值，以及quantile数值
print(df.describe())
#散点图
plt.scatter(df['alcohol'], df['quality'])
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Alcohol and Quality')
plt.show()

plt.scatter(df['volatile acidity'], df['quality'])
plt.xlabel('Volatile Acidity')
plt.ylabel('Quality')
plt.title('Volatile Acidity and Quality')
plt.show()