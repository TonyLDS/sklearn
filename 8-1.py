# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 16:13:13 2017

@author: luzhangqin
"""

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname = r'/usr/share/fonts/truetype/arphic/ukai.ttc')
import numpy as np

X = np.array([
    [0.2, 0.1],
    [0.4, 0.6],
    [0.5, 0.2],
    [0.7, 0.9]
])

y = [0, 0, 0,  1]

markers = ['.', 'x']
plt.scatter(X[:3, 0], X[:3, 1], marker = '.', s =400)
plt.scatter(X[3, 0], X[3, 1], marker = 'x', s= 400)
plt.xlabel('用来睡觉的天数的比例', fontproperties = font)
plt.xlabel('闹脾气的天数比例', fontproperties = font)
plt.title('幼猫和成年猫', fontproperties = font)
plt.show()
