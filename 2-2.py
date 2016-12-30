# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 21:13:30 2016

@author: luzhangqin
"""

import matplotlib.pylab as plt
from matplotlib.font_manager import FontProperties

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
    model = LinearRegression()
    x = [[6], [8], [10], [14], [18]]
    y = [[7], [9], [13], [17.5], [18]]
    
    model.fit(x,y)    
    y2 = model.predict(x2)
    
    x2 = [[0], [10], [14], [25]] 
    y3 = [14.25, 14.25, 14.25, 14.25]
    y4 = y2 * 0.5 + 5
    model.fit(x[1: -1],y[1: -1])
    y5 = model.predict(x2)
    plt.plot(x ,y, 'k.')
    plt.plot(x2, y2, 'g-.')
    plt.plot(x2 ,y3, 'r-.')
    plt.plot(x2 ,y4, 'o-.')
    plt.show()
    
    #残差预测值
    yr = model.predict(x)
    for idx, x in enumerate(x):
        plt.plot([x, x], [y[idx], yr[idx]], 'r-')