#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/12 18:05
# @Author  : mchenyuxiang
# @Email   : mchenyuxiang@foxmail.com
# @Site    : 
# @File    : sgd_test.py
# @Software: PyCharm

import numpy as np
import matrix.mc.SGD as SGD

def sgd_test(user_rank_matrix,opts):
    N = len(user_rank_matrix)
    M = len(user_rank_matrix[0])
    K = opts['rank']
    P = np.random.rand(N, K)
    Q = np.random.rand(M, K)
    nP, nQ = SGD.SGD(user_rank_matrix, P, Q, K,opts['step'],opts['alpha'],opts['beta'])
    direct_sgd_mc = np.dot(nP, nQ.T)
    return direct_sgd_mc
