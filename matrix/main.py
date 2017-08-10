# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 01:10:35 2017

@author: yuxiang
"""
import numpy as np
import matrix.util.HandleFile as HF
import matrix.split.lsh.LSH as LSH
import matrix.mc.SGD as SGD
import matrix.evaluate.Evaluate as EVA

if __name__ == "__main__":
    file_url = 'dataset/ml-100k/ua.base'
    item_url = 'dataset/ml-100k/u.item'
    test_url = 'dataset/ml-100k/ua.test'
    user_size = 943
    item_size = 1682
    style_size = 19

    ## 建立用户兴趣归一矩阵
    user_test_rank_matrix = HF.create_matrix(test_url, user_size, item_size, "\t")
    user_rank_matrix = HF.create_matrix(file_url, user_size, item_size, "\t")
    item_style_matrix = HF.create_file_style_matrix(item_url, item_size, style_size, "|")
    user_style_matrix = HF.create_user_style_matrix(file_url, item_style_matrix, user_size, style_size, "\t")

    ## 将兴趣归一矩阵利用LSH哈希函数将矩阵分块
    k_number = 10
    w = 1
    combin_number = 4
    b = np.random.uniform(0, w)
    test_lsh_index = LSH.lsh_bucket(user_style_matrix, k_number, w, b)
    lsh_split = LSH.lsh_bucket_split(test_lsh_index)
    print(lsh_split)
    # print(test_lsh_index)
    re_test_matrix = LSH.rebuild_matrix(user_rank_matrix, test_lsh_index)
    combin_split = LSH.combine_lsh_bucket(lsh_split, combin_number)
    print(combin_split)
    loc = combin_split.astype(np.int)
    for i in range(len(combin_split)):
        # if loc[i,1] == loc[i,2]:
        #     R = re_test_matrix[loc[i,1],:]
        #     result = np.ones((1,len(R))) * 2
        # else:
        R = re_test_matrix[loc[i, 1]:(loc[i, 2] + 1), :]
        N = len(R)
        M = len(R[0])
        K = 10
        P = np.random.rand(N, K)
        Q = np.random.rand(M, K)
        nP, nQ = SGD.SGD(R, P, Q, K)
        result = np.dot(nP, nQ.T)
        if i == 0:
            final_matrix_temp = result
        else:
            final_matrix_temp = np.vstack((final_matrix_temp, result[(lsh_split.astype(np.int)[i + combin_number-1, 1]-lsh_split.astype(np.int)[i, 1]):(lsh_split.astype(np.int)[i + combin_number-1, 2]+1-lsh_split.astype(np.int)[i, 1]),:]))
            last_center = (lsh_split[i - 1, 0] + w * combin_number) / 2  # 上一轮中心点
            this_center = (lsh_split[i, 0] + w * combin_number) / 2  # 本轮中心点
            distance_split_matrix = test_lsh_index[
                                    lsh_split.astype(np.int)[i, 1]:(lsh_split.astype(np.int)[i + combin_number - 1, 1]),
                                    :]  # 提取重复计算行数
            last_distance = distance_split_matrix[:, 1] - last_center
            this_distance = distance_split_matrix[:, 1] - this_center
            last_result = final_matrix_temp[
                          lsh_split.astype(np.int)[i, 1]:(lsh_split.astype(np.int)[i + combin_number - 1, 1]),
                          :]  # 上一轮计算出的结果
            this_result = result[
                          0:(lsh_split.astype(np.int)[i + combin_number - 1, 1] - lsh_split.astype(np.int)[i, 1]),
                          :]  # 本轮计算结果
            last_weight_temp = (last_distance / (last_distance + this_distance))
            this_weight_temp = (this_distance / (last_distance + this_distance))
            last_weight = np.ones(last_result.shape)
            this_weight = np.ones(this_result.shape)
            for j in range(len(last_weight_temp)):
                last_weight[j] = last_weight_temp[j] * last_weight[j]
                this_weight[j] = this_weight_temp[j] * this_weight[j]
            new_result = last_result * last_weight + this_result * this_weight
            result[
            0:(lsh_split.astype(np.int)[i + combin_number - 1, 1] - lsh_split.astype(np.int)[i, 1]),
            :] = new_result
            final_matrix_temp = np.vstack((final_matrix_temp[
                                           0:lsh_split.astype(np.int)[i - 1, 2] + 1,
                                           :], result))
        print("times:%d" % i)
    # print(lsh_split)
    # print(re_test_matrix)
    # print(test_lsh_index.shape[0])
    final_matrix = LSH.restore_matrix(final_matrix_temp, test_lsh_index)
    # print(final_matrix)

    # 直接用SGD方法
    # N = len(user_rank_matrix)
    # M = len(user_rank_matrix[0])
    # K = 10
    # P = np.random.rand(N, K)
    # Q = np.random.rand(M, K)
    # nP, nQ = SGD.SGD(user_rank_matrix, P, Q, K)
    # direct_sgd_mc = np.dot(nP, nQ.T)
    ## 评价
    lsh_test_error = EVA.test_error(final_matrix, user_test_rank_matrix)
    # direct_test_error = EVA.test_error(direct_sgd_mc, user_test_rank_matrix)
    print("lsh:%f\n" % lsh_test_error)
    # print("sgd:%f\n" % direct_test_error)
