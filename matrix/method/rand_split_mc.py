#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/23 9:12
# @Author  : mchenyuxiang
# @Email   : mchenyuxiang@foxmail.com
# @Site    : 
# @File    : rand_split_mc.py
# @Software: PyCharm

import numpy as np
import matrix.sample.line_sample as line_sample
import matrix.split.lsh.LSH as LSH
import matrix.test_dateset.sgd_test as sgd_test

def split_mc(matrix, P, Q, opts):
    row_number = matrix.shape[0]
    index_squence = line_sample.line_rand_matrix(matrix)
    matrix_squence = LSH.rebuild_matrix(matrix,index_squence)
    P_squence = LSH.rebuild_matrix(P,index_squence)
    split = opts['split_number']
    split_row = np.floor(row_number/split)
    restore_matrix = np.array(matrix)
    for i in range(split):
        if i == split - 1:
            R = matrix_squence[int(i * split_row):int(row_number),:]
            P_R = P_squence[int(i * split_row):int(row_number),:]
            direct_sgd_mc = sgd_test.sgd_test(R,P_R,Q,opts)
            restore_matrix[int(i * split_row):int(row_number),:] = direct_sgd_mc
        else:
            R = matrix_squence[int(i*split_row):int((i+1)*split_row),:]
            P_R = P_squence[int(i*split_row):int((i+1)*split_row),:]
            direct_sgd_mc = sgd_test.sgd_test(R, P_R, Q, opts)
            restore_matrix[int(i*split_row):int((i+1)*split_row),:] = direct_sgd_mc
    final_matrix = LSH.restore_matrix(restore_matrix,index_squence)
    return final_matrix