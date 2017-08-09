#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/9 0:48
# @Author  : mchenyuxiang
# @Email   : mchenyuxiang@foxmail.com
# @Site    : 
# @File    : LSH.py
# @Software: PyCharm
import numpy as np

## 生成桶索引
def lsh_bucket(matrix,k_number,w,b):
    col_size = matrix.shape[1]
    a = np.zeros((k_number,col_size))
    for i in range(k_number):
        a[i] = np.random.randn(1,col_size)

    lsh_index = np.zeros((matrix.shape[0],3))
    for i in range(matrix.shape[0]):
        temp_sum = 0
        temp_sum_ori = 0
        for j in range(k_number):
            temp_sum = temp_sum +  int((np.dot(matrix[i],a[j]) + b)/w)
            temp_sum_ori = temp_sum_ori + (np.dot(matrix[i],a[j]) + b)/w
        temp_sum = temp_sum / k_number
        temp_sum_ori = temp_sum_ori / k_number
        lsh_index[i,0] = temp_sum
        lsh_index[i,1] = temp_sum_ori
        lsh_index[i,2] = i

    lsh_index_arg = np.argsort(lsh_index[:,0]) #按照桶号从下到大排序
    lsh_index = lsh_index[lsh_index_arg]
    return lsh_index

## 重建矩阵
def rebuild_matrix(matrix,lsh_index):
    row_index = matrix.shape[0]
    col_index = matrix.shape[1]
    # re_matrix = np.zeros((row_index,col_index))
    loc = lsh_index.astype(np.int)[:,2]

    re_matrix = matrix[loc]
    return re_matrix

## 根据lsh哈希桶的情况，将矩阵切片
def lsh_bucket_split(lsh_index):
    lsh_matrix_split = np.array([(-99999,0,0)])
    row_index = lsh_index.shape[0]
    # col_index = lsh_index.shape[1]
    temp_split = np.zeros((1,3))
    for i in range(row_index):
        if i == 0:
            temp_split[0,0] = lsh_index[i,0]
            temp_split[0,1] = i
            temp_split[0,2] = i
        elif i < row_index-1:
            if lsh_index[i,0] == lsh_index[i-1,0]:
                continue
            else:
                temp_split[0,2] = i-1
                if lsh_matrix_split[0,0] == -99999:
                    lsh_matrix_split[0, 0] = temp_split[0, 0]
                    lsh_matrix_split[0, 1] = temp_split[0, 1]
                    lsh_matrix_split[0, 2] = temp_split[0, 2]
                else:
                    lsh_matrix_split = np.row_stack((lsh_matrix_split,temp_split))
                temp_split = np.zeros((1,3))
                temp_split[0,0] = lsh_index[i,0]
                temp_split[0,1] = i
                temp_split[0,2] = i
        else:
            if lsh_index[i, 0] == lsh_index[i - 1, 0]:
                temp_split[0, 2] = i
                lsh_matrix_split = np.row_stack((lsh_matrix_split, temp_split))
            else:
                temp_split[0, 2] = i - 1
                lsh_matrix_split = np.row_stack((lsh_matrix_split, temp_split))
                temp_split = np.zeros((1, 3))
                temp_split[0, 0] = lsh_index[i, 0]
                temp_split[0, 1] = i
                temp_split[0, 2] = i
                temp_split[0, 2] = i
                lsh_matrix_split = np.row_stack((lsh_matrix_split, temp_split))
    return lsh_matrix_split



if __name__ ==  "__main__":
    test_matrix = np.array([(1,2,3,4),(1,1,1,1),(1,1,1,1),(1,2,3,4),(3,4,5,6),(5,6,7,8),(3,4,5,6),(5,6,7,8)])
    k_number = 10
    w = 1
    b = np.random.uniform(0,w)
    test_lsh_index = lsh_bucket(test_matrix,k_number,w,b)
    print(test_lsh_index)
    re_test_matrix = rebuild_matrix(test_matrix,test_lsh_index)
    print(re_test_matrix)
    # print(test_lsh_index.shape[0])
    lsh_split = lsh_bucket_split(test_lsh_index)
    print(lsh_split)
