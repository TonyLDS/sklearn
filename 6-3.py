# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:10:55 2017

@author: luzhangqin
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname = r'/usr/share/fonts/truetype/arphic/ukai.ttc', size = 10)

plt.figure(figsize = (8,10))
#字图的位置
plt.subplot(3, 2, 1)
x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
X = np.array(list(zip(x1,x2))).reshape(len(x1), 2)
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('样本', fontproperties = font)
plt.scatter(x1, x2)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']

tests = [2,3,4,5,8]
subplot_counter = 1
for t in tests:
    subplot_counter += 1
    plt.subplot(3, 2, subplot_counter)
    kmeans_model = KMeans(n_clusters = t).fit(X)
    #labels_聚类标签
    for i, l in enumerate(kmeans_model.labels_):
        plt.plot(x1[i], x2[i], color = colors[l], marker = markers[l], ls = 'None')
        plt.xlim([0, 10])
        plt.ylim([0, 10])
        #计算所有样本的平均轮廓系数。
        plt.title('K = %s, 轮廓系数 = %.3f' % (t, metrics.silhouette_score(X, kmeans_model.labels_, metric = 'euclidean')), 
                                           fontproperties = font)