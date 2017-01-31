# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 23:24:58 2017

@author: luzhangqin
"""

import numpy as np
#使用skimage.feature的corner_harris获取harris角点（输入图像应为灰度图像），
#以及corner_peaks对角点进行过滤
from skimage.feature import corner_harris, corner_peaks
#使用skimage.color.rgb2gray进行颜色转换
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
#使用skimge.io读入图像，显示图像
import skimage.io as io
#使用skimage.exposure.equalize_hist进行直方图均衡化
from skimage.exposure import equalize_hist

def show_corners(corners, image):
    fig = plt.figure()
    plt.gray()
    plt.imshow(image)
    #在运行zip(*xyz)之前，xyz的值是：[(1, 4, 7), (2, 5, 8), (3, 6, 9)]
    #那么，zip(*xyz) 等价于 zip((1, 4, 7), (2, 5, 8), (3, 6, 9))
    y_corner, x_corner = zip(*corners)
    plt.plot(x_corner, y_corner, 'or')
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    #指定matplotlib输出图片的尺寸
    fig.set_size_inches(np.array(fig.get_size_inches()) * 1.5)
    plt.show()
    

mandrill =io.imread('mandrill.png')
mandrill = equalize_hist(rgb2gray(mandrill))
corners = corner_peaks(corner_harris(mandrill), min_distance = 2)
show_corners(corners, mandrill)