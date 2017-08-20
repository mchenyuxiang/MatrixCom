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
    row_number = user_style_matrix.shape[0]
    col_number = user_style_matrix.shape[1]
    ## 行拆分
    test_lsh_index = LSH.lsh_new_bucket(user_style_matrix, opts['k_number'])
    # lsh_split = LSH.lsh_bucket_split(test_lsh_index)
    # lsh_split = LSH.lsh_bucket_direct_split(test_lsh_index,split_number=opts['split_number']) # 原始桶
    lsh_split_duplicate = LSH.lsh_bucket_direct_split(test_lsh_index,split_number=opts['split_number'],ratio=opts['ratio'])

    ## 列拆分
    user_style_matrix_temp = user_style_matrix.T
    test_col_index = LSH.lsh_new_bucket(user_style_matrix_temp,opts['k_number'])
    # lsh_col_split = LSH.lsh_bucket_direct_split(test_col_index,split_number_lsh=opts['split_number'])
    lsh_col_split_duplicate = LSH.lsh_bucket_direct_split(test_col_index,split_number=opts['split_col_number'],ratio=opts['ratio'])

    # print(lsh_split)
    # print(test_lsh_index)
    # print(lsh_split_new)
    # print(lsh_split_duplicate)
    # 行重排
    re_row_test_matrix = LSH.rebuild_matrix(user_rank_matrix, test_lsh_index)
    # 列重排
    re_col_matrix_temp = LSH.rebuild_matrix(re_row_test_matrix.T, test_col_index)
    # 重排后复原
    re_test_matrix = re_col_matrix_temp.T
    re_P = LSH.rebuild_matrix(P,test_lsh_index)
    re_Q = LSH.rebuild_matrix(Q,test_col_index)
    # combin_split = LSH.combine_lsh_bucket(lsh_split, combin_number)
    # split_number_lsh = LSH.split_lsh_bucket(lsh_split, opts['combin_number'], opts['split_number'])
    split_number_lsh = lsh_split_duplicate
    split_col_number_lsh = lsh_col_split_duplicate
    # print(split_number_lsh)
    loc = split_number_lsh.astype(np.int)
    loc_col = split_col_number_lsh.astype(np.int)

    ## 将每个块的矩阵记录下来
    result_map = dict()
    for i in range(opts['split_number']):
        for j in range(opts['split_col_number']):
            R = re_test_matrix[loc[i,2]:(loc[i,3]+1),loc_col[j,2]:(loc[j,3]+1)]
            K = opts['rank']
            P_lsh = re_P[loc[i,2]:(loc[i,3]+1),:]
            Q_lsh = re_Q[loc_col[j,2]:(loc[j,3]+1),:]
            nP,nQ = SGD.SGD(R, P_lsh,Q_lsh,K,opts['step'],opts['alpha'],opts['beta'],opts['tol'])
            result = np.dot(nP,nQ.T)
            result_map[str(i)+''+str(j)] = result
            print(i,j)

    # print(result_map)

    ## 对每个块的距离进行计算
    # 计算出每个块的中心点坐标
    sub_matrix = dict()
    for i in range(opts['split_number']):
        for j in range(opts['split_col_number']):
            sub_matrix[str(i)+''+str(j)] = [(split_number_lsh[i,0]+split_number_lsh[i,1])/2,(split_col_number_lsh[j,0]+split_col_number_lsh[j,1])/2]

    print(sub_matrix)
    # 先对行进行计算
    distance_duplicate = dict()
    for i in range(opts['split_number']-1):
        # 行分割
        for j in range(loc[i+1, 2], loc[i, 3]+1):
            for k in range(col_number):
                distance_temp = np.zeros((1, 4))
                for p in range(opts['split_col_number']):
                    if k >= loc_col[p,2] and k <=loc_col[p,3]:
                        middle = sub_matrix[str(i)+''+str(p)]
                        middle_next = sub_matrix[str(i+1)+''+str(p)]
                        break
                d_1 = np.sqrt(np.square((test_lsh_index[j,1]-middle[0])) + np.square((test_col_index[k,1]-middle[1])))
                d_2 = np.sqrt(np.square((test_lsh_index[j,1]-middle_next[0])) + np.square((test_col_index[k,1]-middle_next[1])))
                distance_temp[0,0]=d_1
                distance_temp[0,1]=d_2
                distance_duplicate[str(j)+''+str(k)] = distance_temp

    print("=====================row==============================")

    # 再对列
    for i in range(opts['split_col_number']-1):
        for j in range(loc_col[i+1,2],loc_col[i,3]+1):
            for k in range(row_number):
                if str(k)+''+str(j) in distance_duplicate:
                    distance_temp = distance_duplicate[str(k)+''+str(j)]
                else:
                    distance_temp = np.zeros((1, 4))
                for p in range(opts['split_number']):
                    if k >= loc[p,2] and k <= loc[p,3]:
                        middle = sub_matrix[str(p)+''+str(i)]
                        middle_next = sub_matrix[str(p)+''+str(i+1)]
                        break
                d_3 = np.sqrt(np.square(test_lsh_index[k,1]-middle[0])+np.square(test_col_index[j,1]-middle[1]))
                d_4 = np.sqrt(np.square(test_lsh_index[k,1]-middle_next[0])+np.square(test_col_index[j,1]-middle_next[1]))
                distance_temp[0,2] = d_3
                distance_temp[0,3] = d_4
                distance_duplicate[str(k)+''+str(j)] = distance_temp

    print("=====================col==============================")
    # print(distance_duplicate)

    # 将每个块中的重复计算的数据用权重重新赋值
    for i in range(opts['split_number']-1):
        for j in range(loc[i+1,2],loc[i,3]):
            for k in range(col_number):
                distance_row = distance_duplicate[str(j)+''+str(k)]
                for p in range(opts['split_col_number']):
                    if k >= loc_col[p,2] and k <= loc_col[p,3]:
                        block_matrix_1 = result_map[str(i) + '' + str(p)]
                        block_matrix_2 = result_map[str(i+1) + '' + str(p)]
                        col_split_number = p
                        break
                if distance_row[0,2] == 0 or distance_row[0,3] == 0:
                    weight_1 = (1/distance_row[0,0])/(1/distance_row[0,0]+1/distance_row[0,1])
                    weight_2 = (1/distance_row[0,1])/(1/distance_row[0,0]+1/distance_row[0,1])
                else:
                    weight_1 = (1/distance_row[0,0])/(1/distance_row[0,0]+1/distance_row[0,1]+1/distance_row[0,2]+1/distance_row[0,3])
                    weight_2 = (1/distance_row[0,1])/(1/distance_row[0,0]+1/distance_row[0,1]+1/distance_row[0,2]+1/distance_row[0,3])
                print((j-loc[i,2]),(k-loc_col[col_split_number,2]))
                block_matrix_1[(j-loc[i,2]),(k-loc_col[col_split_number,2])] = weight_1 * block_matrix_1[(j-loc[i,2]),(k-loc_col[col_split_number,2])]
                block_matrix_2[(j-loc[i+1,2]),(k-loc_col[col_split_number,2])] = weight_2 * block_matrix_2[(j-loc[i+1,2]),(k-loc_col[col_split_number,2])]
                result_map[str(i)+''+str(col_split_number)] = block_matrix_1
                result_map[str(i+1)+''+str(col_split_number)] = block_matrix_2
    print("=====================row==============================")

    for i in range(opts['split_col_number']-1):
        for j in range(loc_col[i+1,2],loc_col[i,3]+1):
            for k in range(row_number):
                distance_row = distance_duplicate[str(k)+''+str(j)]
                for p in range(opts['split_number']):
                    if k >= loc[p,2] and k <= loc[p,3]:
                        block_matrix_1 = result_map[str(p) + '' + str(i)]
                        block_matrix_2 = result_map[str(p) + '' + str(i+1)]
                        col_split_number = p
                        break
                if distance_row[0,0] == 0 or distance_row[0,1] == 0:
                    weight_1 = (1/distance_row[0,2])/(1/distance_row[0,2]+1/distance_row[0,3])
                    weight_2 = (1/distance_row[0,3])/(1/distance_row[0,2]+1/distance_row[0,3])
                else:
                    weight_1 = (1/distance_row[0,2])/(1/distance_row[0,0]+1/distance_row[0,1]+1/distance_row[0,2]+1/distance_row[0,3])
                    weight_2 = (1/distance_row[0,3])/(1/distance_row[0,0]+1/distance_row[0,1]+1/distance_row[0,2]+1/distance_row[0,3])

                block_matrix_1[(k-loc[col_split_number,2]),(j-loc_col[i,2])] = weight_1 * block_matrix_1[(k-loc[col_split_number,2]),(j-loc_col[i,2])]
                block_matrix_2[(k-loc[col_split_number,2]),(j-loc_col[i+1,2])] = weight_2 * block_matrix_2[(k-loc[col_split_number,2]),(j-loc_col[i+1,2])]
                result_map[str(col_split_number)+''+str(i)] = block_matrix_1
                result_map[str(col_split_number)+''+str(i+1)] = block_matrix_2

    print("=====================col==============================")

    ## 矩阵合并
    final_matrix_temp = np.zeros(user_style_matrix.shape)
    for i in range(opts['split_number']):
        for j in range(opts['split_col_number']):
            result_map_temp = result_map[str(i)+''+str(j)]
            for p in range(loc[i,2],loc[i,3]+1):
                for q in range(loc_col[j,2],loc_col[j,3]+1):
                    final_matrix_temp[p,q] = final_matrix_temp[p,q]+result_map_temp[p-loc[i,2],q-loc_col[j,2]]

    # print(lsh_split)
    # print(re_test_matrix)
    # print(test_lsh_index.shape[0])
    final_row_matrix= LSH.restore_matrix(final_matrix_temp, test_lsh_index)
    final_row_matrix_temp = final_row_matrix.T
    final_col_matrix=LSH.restore_matrix(final_row_matrix_temp, test_col_index)
    final_matrix = final_col_matrix.T
    return final_matrix
    # print(final_matrix)

## 对行进行计算
def row_mc(user_rank_matrix,user_style_matrix,P,Q,opts):
    ## 行拆分
    test_lsh_index = LSH.lsh_new_bucket(user_style_matrix, opts['k_number'])
    # lsh_split = LSH.lsh_bucket_split(test_lsh_index)
    # lsh_split = LSH.lsh_bucket_direct_split(test_lsh_index, split_number=opts['split_number'])  # 原始桶
    lsh_split_duplicate = LSH.lsh_bucket_direct_split(test_lsh_index, split_number=opts['split_number'],
                                                      ratio=opts['ratio'])

    # ## 列拆分
    # user_style_matrix_temp = user_style_matrix.T
    # test_col_index = LSH.lsh_new_bucket(user_style_matrix_temp, opts['k_number'])
    # # lsh_col_split = LSH.lsh_bucket_direct_split(test_col_index,split_number_lsh=opts['split_number'])
    # lsh_col_split_duplicate = LSH.lsh_bucket_direct_split(test_col_index, split_number=opts['split_col_number'],
    #                                                       ratio=opts['ratio'])

    # print(lsh_split)
    # print(test_lsh_index)
    # print(lsh_split_new)
    # print(lsh_split_duplicate)
    # 行重排
    re_row_test_matrix = LSH.rebuild_matrix(user_rank_matrix, test_lsh_index)
    # 列重排
    # re_col_matrix_temp = LSH.rebuild_matrix(re_row_test_matrix.T, test_col_index)
    # 重排后复原
    re_test_matrix = re_row_test_matrix
    re_P = LSH.rebuild_matrix(P, test_lsh_index)
    # re_Q = LSH.rebuild_matrix(Q, test_col_index)
    # combin_split = LSH.combine_lsh_bucket(lsh_split, combin_number)
    # split_number_lsh = LSH.split_lsh_bucket(lsh_split, opts['combin_number'], opts['split_number'])
    split_number_lsh = lsh_split_duplicate
    # print(split_number_lsh)
    loc = split_number_lsh.astype(np.int)
    for i in range(opts['split_number']):
        # if loc[i,1] == loc[i,2]:
        #     R = re_test_matrix[loc[i,1],:]
        #     result = np.ones((1,len(R))) * 2
        # else:
        R = re_test_matrix[loc[i, 2]:(loc[i, 3] + 1), :]
        # N = len(R)
        # M = len(R[0])
        K = opts['rank']
        P_lsh = re_P[loc[i, 2]:(loc[i, 3] + 1), :]
        # Q = np.random.rand(M, K)
        nP, nQ = SGD.SGD(R, P_lsh, Q, K, opts['step'], opts['alpha'], opts['beta'], opts['tol'])
        print("========================================")
        result = np.dot(nP, nQ.T)
        if i == 0:
            final_matrix_temp = result
        else:
            loc_split = split_number_lsh.astype(np.int)
            final_matrix_temp = np.vstack(
                (final_matrix_temp[0:loc_split[i, 2], :], result[0:(loc_split[i, 3] + 1 - loc_split[i, 2]), :]))
            last_center = (lsh_split_duplicate[i - 1, 0] + lsh_split_duplicate[i - 1, 1]) / 2  # 上一轮中心点
            this_center = (lsh_split_duplicate[i, 0] + lsh_split_duplicate[i, 1]) / 2  # 本轮中心点
            distance_split_matrix = test_lsh_index[loc_split[i, 2]:(loc_split[i - 1, 3] + 1), :]  # 提取重复计算行数
            last_distance = distance_split_matrix[:, 1] - last_center
            this_distance = distance_split_matrix[:, 1] - this_center
            last_result = final_matrix_temp[loc_split[i, 2]:(loc_split[i - 1, 3] + 1), :]  # 上一轮计算出的结果
            this_result = result[0:loc_split[i - 1, 3] - loc_split[i, 2] + 1, :]  # 本轮计算结果
            last_weight_temp = ((1 / last_distance) / (1 / last_distance + 1 / this_distance))
            this_weight_temp = ((1 / this_distance) / (1 / last_distance + 1 / this_distance))
            last_weight = np.ones(last_result.shape)
            this_weight = np.ones(this_result.shape)
            for j in range(len(last_weight_temp)):
                last_weight[j] = last_weight_temp[j] * last_weight[j]
                this_weight[j] = this_weight_temp[j] * this_weight[j]
            new_result = last_result * last_weight + this_result * this_weight
            result[0:loc_split[i - 1, 3] - loc_split[i, 2] + 1, :] = new_result
            final_matrix_temp = np.vstack((final_matrix_temp[0:loc_split[i, 2], :], result))
            # print("times:%d" % i)
    # print(lsh_split)
    # print(re_test_matrix)
    # print(test_lsh_index.shape[0])
    final_row_matrix = LSH.restore_matrix(final_matrix_temp, test_lsh_index)
    final_matrix = final_row_matrix
    # final_col_matrix = LSH.restore_matrix(final_row_matrix_temp, test_col_index)
    # final_matrix = final_col_matrix.T
    return final_matrix
    # print(final_matrix)