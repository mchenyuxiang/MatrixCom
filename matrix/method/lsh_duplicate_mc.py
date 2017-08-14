#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/12 17:39
# @Author  : mchenyuxiang
# @Email   : mchenyuxiang@foxmail.com
# @Site    : 
# @File    : lsh_duplicate_mc.py
# @Software: PyCharm

import numpy as np
import matrix.split.lsh.LSH as LSH
import matrix.mc.SGD as SGD

def lsh_mc(user_rank_matrix,user_style_matrix,P,Q,opts):
    # k_number = 10
    # w = 0.9
    # combin_number = 2
    # split_number = 2
    # b = np.random.uniform(0, opts['w'])
    test_lsh_index = LSH.lsh_bucket(user_style_matrix, opts['k_number'], opts['w'], opts['b'])
    lsh_split = LSH.lsh_bucket_split(test_lsh_index)
    print(lsh_split)
    # print(test_lsh_index)
    re_test_matrix = LSH.rebuild_matrix(user_rank_matrix, test_lsh_index)
    re_P = LSH.rebuild_matrix(P,test_lsh_index)
    # combin_split = LSH.combine_lsh_bucket(lsh_split, combin_number)
    split_number_lsh = LSH.split_lsh_bucket(lsh_split, opts['combin_number'], opts['split_number'])
    print(split_number_lsh)
    loc = split_number_lsh.astype(np.int)
    for i in range(opts['split_number']):
        # if loc[i,1] == loc[i,2]:
        #     R = re_test_matrix[loc[i,1],:]
        #     result = np.ones((1,len(R))) * 2
        # else:
        R = re_test_matrix[loc[i, 1]:(loc[i, 2] + 1), :]
        # N = len(R)
        # M = len(R[0])
        K = opts['rank']
        P_lsh = re_P[loc[i, 1]:(loc[i, 2] + 1), :]
        # Q = np.random.rand(M, K)
        nP, nQ = SGD.SGD(R, P_lsh, Q, K,opts['step'],opts['alpha'],opts['beta'])
        result = np.dot(nP, nQ.T)
        if i == 0:
            final_matrix_temp = result
        else:
            loc_split = split_number_lsh.astype(np.int)
            final_matrix_temp = np.vstack(
                (final_matrix_temp[0:loc_split[i, 1], :], result[0:(loc_split[i, 2] + 1 - loc_split[i, 1]), :]))
            last_center = (test_lsh_index[loc[i - 1, 1], 2] + test_lsh_index[loc[i - 1, 1], 1]) / 2  # 上一轮中心点
            this_center = (test_lsh_index[loc[i, 2], 1] + test_lsh_index[loc[i, 1], 1]) / 2  # 本轮中心点
            distance_split_matrix = test_lsh_index[loc_split[i, 1]:(loc_split[i - 1, 2] + 1), :]  # 提取重复计算行数
            last_distance = distance_split_matrix[:, 1] - last_center
            this_distance = distance_split_matrix[:, 1] - this_center
            last_result = final_matrix_temp[loc_split[i, 1]:(loc_split[i - 1, 2] + 1), :]  # 上一轮计算出的结果
            this_result = result[0:loc_split[i - 1, 2] - loc_split[i, 1] + 1, :]  # 本轮计算结果
            last_weight_temp = (last_distance / (last_distance + this_distance))
            this_weight_temp = (this_distance / (last_distance + this_distance))
            last_weight = np.ones(last_result.shape)
            this_weight = np.ones(this_result.shape)
            for j in range(len(last_weight_temp)):
                last_weight[j] = last_weight_temp[j] * last_weight[j]
                this_weight[j] = this_weight_temp[j] * this_weight[j]
            new_result = last_result * last_weight + this_result * this_weight
            result[0:loc_split[i - 1, 2] - loc_split[i, 1] + 1, :] = new_result
            final_matrix_temp = np.vstack((final_matrix_temp[0:loc_split[i, 1], :], result))
        print("times:%d" % i)
    # print(lsh_split)
    # print(re_test_matrix)
    # print(test_lsh_index.shape[0])
    final_matrix = LSH.restore_matrix(final_matrix_temp, test_lsh_index)
    return final_matrix
    # print(final_matrix)