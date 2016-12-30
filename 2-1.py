# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 17:50:47 2016

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
    x2 = [[0], [10], [14], [25]] 
    #.用点画图,k是颜色
    #plt.plot(x, y, 'k.')
    #plt.show()
    
    #创建拟合模型
    model = LinearRegression()
    #模型训练
    model.fit(x,y)
    #模型预测
    #model.predict([a][b]) [a]为x，[b]返回第几个x的值
    print('预测一张12英寸的批萨价格: $%.2f' %model.predict([12][0]))
    
    y2 = model.predict(x2)
    plt.plot(x, y, 'k.')
    plt.plot(x2, y2, 'g-')
    plt.show()
    
    
    