# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 11:44:39 2016

@author: luzhangqin
"""

import matplotlib.pylab as plt
from matplotlib.font_manager import FontProperties

import numpy as np

from sklearn.linear_model import LinearRegression

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

if __name__ == '__main__':
    plt = runplt()
    x = [[6], [8], [10], [14], [18]]
    y = [[7], [9], [13], [17.5], [18]]
    x2 = [[0], [10], [14], [25]]
    
    plt.plot(x, y, 'k.')
    
    model = LinearRegression()
    model.fit(x,y)    
    y2 = model.predict(x2)
    plt.plot(x2, y2, 'g-')
    
    #残差预测值
    yr = model.predict(x)
    for idx, xi in enumerate(x):
        plt.plot([xi, xi], [y[idx], yr[idx]], 'r-')
    
    plt.show()
    
    #残差平方和
    #np.array(x).reshape(1, -1) list->array
    print('残差平方和： %.2f' 
        %np.mean((model.predict(x) - y)**2))