# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 14:01:11 2017

@author: luzhangqin
"""

import numpy as np
from sklearn.cluster import KMeans 
from sklearn.utils import shuffle
#图像处理库
import mahotas as mh
import matplotlib.pyplot as plt


original_img = np.array(mh.imread('timg.jpeg'), dtype = np.float64) / 255
width, height, depth = tuple(original_img.shape)
image_flattened = np.reshape(original_img, (width * height, depth))
image_array_sample = shuffle(image_flattened, random_state = 0)[:1000]
estimator = KMeans(n_clusters = 64, random_state = 0)
estimator.fit(image_array_sample)
cluster_assignments = estimator.predict(image_flattened)
compressed_palette = estimator.cluster_centers_
#返回给定形状和类型的新数组，用零填充。
compressed_img = np.zeros((width, height, compressed_palette.shape[1]))
label_idx = 0
for i in range(width):
    for j in range(height):
        compressed_img[i][j] = compressed_palette[
                cluster_assignments[label_idx]]
        label_idx += 1

#计算所有样本的平均轮廓系数。
#在nrows，ncols和plot_number都小于10的情况下，存在方便，使得可以给出3位数字，
#其中，数百代表nrow，十位代表ncol 并且单位表示plot_number。 例如：
#plt.subplot(1, 2, 2)
plt.subplot(122)
plt.title('original_img')
plt.imshow(original_img)
plt.axis('off')

plt.subplot(121)
plt.title('compressed_img')
plt.imshow(compressed_img)
plt.axis('off')
plt.show()