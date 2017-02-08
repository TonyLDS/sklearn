# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 15:32:58 2017

@author: luzhangqin
"""

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
#linux shell: fc-list :lang=zh
font = FontProperties(fname=r'/usr/share/fonts/truetype/arphic/ukai.ttc', size = 10)
import numpy as np
plt.figure()
plt.axis([-6, 6, 0, 1])
plt.grid(True)
X = np.arange(-6, 6, 0.1)
#np.e = 2.718281828459045
y = 1 / (1 + np.e ** (-X))
plt.plot(X, y, 'b-')