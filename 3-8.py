# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 13:19:10 2017

@author: luzhangqin
"""

import mahotas as mh
from mahotas.features import surf
#surf
image = mh.imread('timg.jpeg', as_grey = True)
print('第一个SURF描述符：\n{}\n'.format(surf.surf(image)[0]))
print('抽取了%s个SURF描述符' %(len(surf.surf(image))))