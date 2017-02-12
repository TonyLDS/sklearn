# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 20:02:39 2017

@author: luzhangqin
"""

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r'/usr/share/fonts/truetype/arphic/ukai.ttc', size = 10)


y_test = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 1, 0, 0, 0, 0, 0, 1, 1, 1]
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
plt.matshow(confusion_matrix)
plt.title('混淆矩阵', fontproperties = font)
#图例
plt.colorbar()
plt.ylabel('实际类型', fontproperties = font)
plt.xlabel('预测类型', fontproperties = font)
plt.show()

#准确率
from sklearn.metrics import accuracy_score
y_pred, y_true = [0, 1, 1, 0], [1, 1, 1, 1]
print(accuracy_score(y_true, y_pred))