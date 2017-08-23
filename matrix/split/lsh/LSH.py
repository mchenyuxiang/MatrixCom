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
def lsh_bucket(matrix, k_number, w, b):
    col_size = matrix.shape[1]
    a = np.zeros((k_number, col_size))
    row_size = matrix.shape[0]
    for i in range(row_size):
        matrix[i] = matrix[i] / np.sum(matrix[i])

    for i in range(k_number):
        a[i] = np.random.randn(1, col_size)

    lsh_index = np.zeros((matrix.shape[0], 3))
    for i in range(matrix.shape[0]):
        temp_sum = 0
        temp_sum_ori = 0
        for j in range(k_number):
            temp_sum = temp_sum + (np.dot(matrix[i], a[j]) + b)
            temp_sum_ori = temp_sum_ori + (np.dot(matrix[i], a[j]) + b)
        temp_sum = int((temp_sum / k_number) / w)
        temp_sum_ori = (temp_sum_ori / k_number) / w
        lsh_index[i, 0] = temp_sum
        lsh_index[i, 1] = temp_sum_ori
        lsh_index[i, 2] = i

    lsh_index_arg = np.argsort(lsh_index[:, 1])  # 按照桶号从下到大排序
    lsh_index = lsh_index[lsh_index_arg]
    return lsh_index


def lsh_new_bucket(matrix, k_number):
    col_size = matrix.shape[1]
    a = np.zeros((k_number, col_size))
    row_size = matrix.shape[0]

    matrix_ori = np.array(matrix)
    ## 对原始矩阵进行归一
    for i in range(row_size):
        matrix_sum = np.sum(matrix_ori[i])
        if matrix_sum == 0:
            continue
        matrix_ori[i] = matrix_ori[i] / matrix_sum

    ## 生成hash函数组
    for i in range(k_number):
        a[i] = np.random.randn(1, col_size)

    lsh_index = np.zeros((matrix.shape[0], 3))
    for i in range(matrix.shape[0]):
        temp_sum = 0
        temp_sum_ori = 0
        for j in range(k_number):
            temp_sum = temp_sum + np.dot(matrix_ori[i], a[j])
            temp_sum_ori = temp_sum_ori + temp_sum
        # print(i,j,temp_sum)
        temp_sum = int((temp_sum / k_number))
        temp_sum_ori = (temp_sum_ori / k_number)
        lsh_index[i, 0] = temp_sum
        lsh_index[i, 1] = temp_sum_ori
        lsh_index[i, 2] = i

    lsh_index_arg = np.argsort(lsh_index[:, 1])  # 按照桶号从小到大排序
    lsh_index = lsh_index[lsh_index_arg]
    return lsh_index


## 重建矩阵
def rebuild_matrix(matrix, lsh_index):
    # row_index = matrix.shape[0]
    # col_index = matrix.shape[1]
    # re_matrix = np.zeros((row_index,col_index))
    loc = lsh_index.astype(np.int)[:, 2]

    re_matrix = matrix[loc]
    return re_matrix


## 根据半径来进行矩阵划分
## lsh_index 中，第一列为最小坐标，第二列为最大坐标
def lsh_bucket_direct_split(lsh_index, split_number=2, ratio=1):
    lsh_matrix_split = np.zeros((split_number, 4))
    row_number = len(lsh_index)
    max_index = np.max(lsh_index[:, 1])
    min_index = np.min(lsh_index[:, 1])
    distance = max_index - min_index
    block = distance / split_number
    for i in range(split_number):
        middle_bucket = (2 * lsh_index[0, 1] + block * (i + 1) + block * i) / 2
        end_flag_temp = middle_bucket - (lsh_index[0, 1] + block * i)
        end_flag = end_flag_temp * ratio
        if i == 0:
            lsh_matrix_split[0, 0] = min_index
            lsh_matrix_split[i, 1] = middle_bucket + end_flag
        elif i == split_number - 1:
            lsh_matrix_split[i, 0] = middle_bucket - end_flag
            lsh_matrix_split[i, 1] = max_index
        else:
            lsh_matrix_split[i, 0] = middle_bucket - end_flag
            lsh_matrix_split[i, 1] = middle_bucket + end_flag

    for i in range(split_number):
        lsh_matrix_split[i, 2] = -1
        for j in range(row_number):
            if (lsh_index[j, 1] > lsh_matrix_split[i, 0] or np.isclose(lsh_index[j, 1], lsh_matrix_split[i, 0])) and (
                    lsh_index[j, 1] < lsh_matrix_split[i, 1] or np.isclose(lsh_index[j, 1], lsh_matrix_split[i, 1])):
                if lsh_matrix_split[i, 2] == -1:
                    lsh_matrix_split[i, 2] = j
                if lsh_matrix_split[i, 2] >= j:
                    lsh_matrix_split[i, 2] = j
                if lsh_matrix_split[i, 3] <= j:
                    lsh_matrix_split[i, 3] = j

    # for i in range(row_number):
    #     if i != 0 and lsh_index[i,1] >= lsh_matrix_split[s_i,0] and lsh_matrix_split[s_i,2] == 0 :
    #         lsh_matrix_split[s_i, 2] = i
    #     if lsh_index[i,1] > lsh_matrix_split[s_i,1]:
    #         s_i = s_i + 1
    #     if s_i == 0:
    #         lsh_matrix_split[s_i,2]=0
    #     lsh_matrix_split[s_i,3]=i

    return lsh_matrix_split


## 根据lsh哈希桶的情况，将矩阵切片
def lsh_bucket_split(lsh_index):
    lsh_matrix_split = np.array([(-99999, 0, 0)])
    row_index = lsh_index.shape[0]
    # col_index = lsh_index.shape[1]
    temp_split = np.zeros((1, 3))
    for i in range(row_index):
        if i == 0:
            temp_split[0, 0] = lsh_index[i, 0]
            temp_split[0, 1] = i
            temp_split[0, 2] = i
        elif i < row_index - 1:
            if lsh_index[i, 0] == lsh_index[i - 1, 0]:
                continue
            else:
                temp_split[0, 2] = i - 1
                if lsh_matrix_split[0, 0] == -99999:
                    lsh_matrix_split[0, 0] = temp_split[0, 0]
                    lsh_matrix_split[0, 1] = temp_split[0, 1]
                    lsh_matrix_split[0, 2] = temp_split[0, 2]
                else:
                    lsh_matrix_split = np.row_stack((lsh_matrix_split, temp_split))
                temp_split = np.zeros((1, 3))
                temp_split[0, 0] = lsh_index[i, 0]
                temp_split[0, 1] = i
                temp_split[0, 2] = i
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


def combine_lsh_bucket(lsh_matrix_split, number):
    row_number = len(lsh_matrix_split)
    combine_lsh = np.zeros((row_number - number + 1, 3))
    for i in range(row_number - number + 1):
        combine_lsh[i, 0] = i
        combine_lsh[i, 1] = lsh_matrix_split[i, 1]
        combine_lsh[i, 2] = lsh_matrix_split[i + number - 1, 2]
    return combine_lsh


## 根据split_number数量均分桶，再在两端加入number个桶
def split_lsh_bucket(lsh_matrix_split, number, split_number):
    row_number = len(lsh_matrix_split)
    split_lsh = np.zeros((split_number, 3))
    k = row_number / split_number
    for i in range(split_number):
        if i == 0:
            split_lsh[i, 0] = i
            split_lsh[i, 1] = lsh_matrix_split[i, 1]
            split_lsh[i, 2] = lsh_matrix_split[int(np.ceil((i + 1) * k) + number - 1), 2]
        elif i == (split_number - 1):
            split_lsh[i, 0] = i
            split_lsh[i, 1] = lsh_matrix_split[int(np.floor(i * k) - number + 1), 1]
            split_lsh[i, 2] = lsh_matrix_split[row_number - 1, 2]
        else:
            split_lsh[i, 0] = i
            split_lsh[i, 1] = lsh_matrix_split[int(np.floor(i * k) - number + 1), 1]
            split_lsh[i, 2] = lsh_matrix_split[int(np.ceil((i + 1) * k) + number - 1), 2]
    return split_lsh


## 还原矩阵
def restore_matrix(matrix, lsh_index):
    final_matrix = np.zeros((matrix.shape[0], matrix.shape[1]))
    row_index = lsh_index.shape[0]
    loc = lsh_index.astype(np.int)[:, 2]
    for i in range(row_index):
        final_matrix[loc[i]] = matrix[i]
    return final_matrix
    pass


if __name__ == "__main__":
    test_matrix = np.array(
        [(1, 2, 3, 4), (1, 1, 1, 1), (1, 1, 1, 1), (1, 2, 3, 5), (3, 4, 5, 6), (5, 6, 7, 8), (3, 4, 5, 6),
         (5, 6, 7, 8)])
    k_number = 10
    w = 1
    b = np.random.uniform(0, w)
    test_lsh_index = lsh_new_bucket(test_matrix, k_number)
    print(test_lsh_index)
    re_test_matrix = rebuild_matrix(test_matrix, test_lsh_index)
    print(re_test_matrix)
    # print(test_lsh_index.shape[0])
    lsh_split = lsh_bucket_direct_split(test_lsh_index, split_number=2, ratio=1)
    print(lsh_split)
    final_matrix = restore_matrix(re_test_matrix, test_lsh_index)
    print(final_matrix)
