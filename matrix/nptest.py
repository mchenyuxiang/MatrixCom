# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 00:15:34 2017

@author: yuxiang
"""

import numpy as np
import matrix.mc.lmafit as lmafit

a = np.arange(15).reshape(3,5)

print(a)
#print(a.shape)
#print(a.itemsize)

U,sigma,VT = np.linalg.svd(a)

#print(U)
#print(sigma)
#print(a.shape[0])

sig = np.zeros((a.shape[0],a.shape[1]))
for i in range(a.shape[0]):
    sig[i,i] = sigma[i]

b=U.dot(sig).dot(VT)
print("==============")
print(b)

bool_idx = np.ones(a.shape)
bool_idx = bool_idx.astype(int)
X,Y,Out = lmafit.lmafit_mc_adp(3,5,5,bool_idx,a,1)
Final = X.dot(Y)