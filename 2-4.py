# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 15:59:31 2016

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
    x = [[6], [8], [10], [14], [18]]
    y = [[7], [9], [13], [17.5], [18]]
    #测试集
    x_test = [[8], [9], [11], [16], [12]]
    y_test = [[11], [8.5], [15], [18], [11]]
    
    model = LinearRegression()
    model.fit(x,y)
    #R方
    r2 = model.score(x_test, y_test)
    print(r2)