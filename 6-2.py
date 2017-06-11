# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:39:58 2017

@author: luzhangqin
"""

import numpy as np
from sklearn.cluster import KMeans
#计算两个输入集合的每对之间的距离。
from scipy.spatial.distance import cdist
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

font = FontProperties(fname = r'/usr/share/fonts/truetype/arphic/ukai.ttc', size = 10)

#输出间隔的下限。 所有生成的值将大于或等于低。 默认值为0。
#t或ints的元组，可选输出形状。 
#如果给定形状是例如（m，n，k），则绘制m * n * k个样本。
#默认值为None，在这种情况下返回单个值。
cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(3.5, 4.5, (2, 10))
#取一个数组序列，并水平堆叠，形成一个数组。
#转置
X = np.hstack((cluster1, cluster2)).T
plt.figure()
plt.axis([0, 5, 0, 5])
plt.grid(True)
plt.plot(X[:, 0], X[:, 1], 'k.')
plt.show()
#取一个数组序列，并将它们垂直堆叠以构成一个数组。
#X = np.vstack((cluster1, cluster2)).T

K = range(1, 10)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(X)
    #kmeans.cluster_centers_ 集群中心的坐标
    #axis如果这是一个int的元组，则选择多个轴的最小值，而不是单个轴或所有轴，如前所述。
    meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 
                                            'euclidean'), axis = 1)) / X.shape[0])
plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('平均畸变程度', fontproperties = font )
plt.title('肘部法则', fontproperties = font )
