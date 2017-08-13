#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/12 23:44
# @Author  : mchenyuxiang
# @Email   : mchenyuxiang@foxmail.com
# @Site    : 
# @File    : line_sample.py
# @Software: PyCharm

import numpy as np

## 按照矩阵大小生成每行的随机采样位置矩阵
def line_sample_squence(matrix):
    row_number = matrix.shape[0]
    col_number = matrix.shape[1]
    sample_squence = np.zeros((row_number,col_number))
    for i in range(row_number):
        sample_squence[i] = np.linspace(0,col_number-1,col_number)
        np.random.shuffle(sample_squence[i])

    return sample_squence

## 生成采样矩阵
def line_sample_matrix(matrix,sample_squence,rate):
    row_number = matrix.shape[0]
    col_number = matrix.shape[1]
    sample_number = int(np.floor(col_number * rate))
    sample_matrix = np.zeros(matrix.shape)
    sample_squence_loc = sample_squence.astype(np.int)
    for i in range(row_number):
        sample_loc = sample_squence_loc[i,0:sample_number]
        sample_matrix[i,sample_loc] = matrix[i,sample_loc]
    return sample_matrix

if __name__ == "__main__":
    a = np.random.random((5,4))
    b = line_sample_squence(a)
    print(b)
    c = line_sample_matrix(a,b,0.5)
    print(c)

