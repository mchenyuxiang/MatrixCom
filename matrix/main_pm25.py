# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 01:10:35 2017

@author: yuxiang
"""
from scipy import io

import numpy as np
import matrix.evaluate.Evaluate as EVA
import matrix.method.lsh_duplicate_row_col_mc as lsh_duplicate_row_col_mc
import matrix.test_dateset.sgd_test as sgd_test
import matrix.sample.line_sample as line_sample
import matrix.method.rand_split_mc as rand_split_test
import datetime

if __name__ == "__main__":
    # 参数设定
    k_number = 10
    w = 0.0001
    combin_number = 2
    split_number = 3
    split_col_number = 1
    b = np.random.uniform(0, w)
    rank = 8
    aplha = 0.00055
    beta = 0.02
    step_lsh = 400
    step_sgd = 400
    rate = 0.8
    tol = 1e-7
    ratio = 1.5

    ## ml-100k 数据集
    # user_test_rank_matrix:    测试矩阵
    # user_rank_matrix:         训练矩阵
    # item_style_matrix:        电影类别矩阵
    # user_style_matrix:        用户电影类别矩阵(局部敏感哈希函数分类矩阵)
    # file_url = 'dataset/ml-100k/ua.base'
    # item_url = 'dataset/ml-100k/u.item'
    # test_url = 'dataset/ml-100k/ua.test'
    # user_test_rank_matrix, user_rank_matrix, item_style_matrix, user_style_matrix = ml100k.test_ml100k(file_url,item_url,test_url)

    ## geant 数据集
    # file_url = 'dataset/gant/GeantOD.mat'
    # save_url = 'dataset/gant'
    # extra_name = 'GeantMatrixODNorm'
    # # extra_name = 'GeantMatrixOD'
    # save_url_name = save_url + "/" + extra_name + ".npy"
    # user_ori_matrix = np.load(save_url_name)[:,0:528].T  # 原始矩阵

    ## pings 数据集
    # save_url = 'dataset/pings_4hr'
    # extra_name = 'originMatrix4h30min'
    # save_url_name = save_url + "/" + extra_name + ".npy"
    # user_ori_matrix = np.load(save_url_name)

    ## pm25 数据集
    save_url = 'dataset/weather'
    extra_name = 'PM_Data'
    save_url_name = save_url + "/" + extra_name + ".npy"
    user_ori_matrix = np.load(save_url_name)

    # extra_name = 'dataset/result/rank_alpha_lsh'
    # extra_name = 'GeantMatrixOD'
    # io_file.save_mat(file_url,save_url,extra_name) # 将mat文件保存为npy文件
    # save_url_name = extra_name + ".npy"
    flag = 1
    while flag == 1:
        for i in range(len(user_ori_matrix)):
            if np.sum(user_ori_matrix[i]) == 0:
                user_ori_matrix = np.delete(user_ori_matrix, i, 0)
                break
        if i == len(user_ori_matrix) - 1:
            flag = 0

    user_ori_matrix = user_ori_matrix / 1000
    # print("==============")

    print("================end load file==============")
    # R = [
    #     [5, 3, 4, 1, 2, 1],
    #     [4, 2, 3, 1, 2, 3],
    #     [1, 1, 5, 5, 2, 5],
    #     [1, 1, 4, 4, 2, 4],
    #     [2, 1, 5, 4, 2, 4],
    #     [1, 1, 5, 5, 2, 5],
    #     [3, 1, 5, 4, 2, 4],
    # ]
    # user_ori_matrix = np.array(R).T
    user_sample_squence = line_sample.line_sample_squence(user_ori_matrix)  # 生成采样序列
    print("================end sample squence==============")

    # print(b)
    user_rank_matrix = line_sample.line_sample_matrix(user_ori_matrix, user_sample_squence, rate)  # 生成训练矩阵
    print("================end sample==============")
    user_style_matrix = user_rank_matrix
    user_test_rank_matrix = user_ori_matrix - user_rank_matrix  # 生成验证结果矩阵
    # print(c)

    result_lsh_rank_matrix = np.zeros(len(np.arange(0.1, 1, 0.1)))
    result_lsh_time_matrix = np.zeros(len(np.arange(0.1, 1, 0.1)))
    result_lsh_nodu_rank_matrix = np.zeros(len(np.arange(0.1, 1, 0.1)))
    result_lsh_nodu_time_matrix = np.zeros(len(np.arange(0.1, 1, 0.1)))
    result_sgd_rank_matrix = np.zeros(len(np.arange(0.1, 1, 0.1)))
    result_sgd_time_matrix = np.zeros(len(np.arange(0.1, 1, 0.1)))
    result_rand_rank_matrix = np.zeros(len(np.arange(0.1, 1, 0.1)))
    result_rand_time_matrix = np.zeros(len(np.arange(0.1, 1, 0.1)))

    i = 0
    for rate in np.arange(0.1, 1, 0.1):
        starttime = datetime.datetime.now()
        # 参数
        opts = {
            'k_number': k_number,  # 哈希桶个数
            'w': w,  # 桶宽
            'combin_number': combin_number,  # 联合桶个数
            'split_number': split_number,  # 行分块个数
            'split_col_number': split_col_number,  # 列分块个数
            'b': b,  # 局部敏感哈希函数中b
            'rank': rank,  # 预估的矩阵的秩
            'alpha': aplha,  # sgd 参数
            'beta': beta,  # sgd 参数
            'step': step_lsh,  # 循环计算次数
            'step_sgd': step_sgd,  # sgd循环次数
            'rate': rate,  # 采样率
            'tol': tol,
            'ratio': ratio,  # 半径改变
        }

        opts_no = {
            'k_number': k_number,  # 哈希桶个数
            'w': w,  # 桶宽
            'combin_number': combin_number,  # 联合桶个数
            'split_number': split_number,  # 行分块个数
            'split_col_number': split_col_number,  # 列分块个数
            'b': b,  # 局部敏感哈希函数中b
            'rank': rank,  # 预估的矩阵的秩
            'alpha': aplha,  # sgd 参数
            'beta': beta,  # sgd 参数
            'step': step_lsh,  # 循环计算次数
            'step_sgd': step_sgd,  # sgd循环次数
            'rate': rate,  # 采样率
            'tol': tol,
            'ratio': 1,  # 半径改变
        }

        # 生成sgd的因子矩阵
        N = len(user_rank_matrix)
        M = len(user_rank_matrix[0])
        K = opts['rank']
        P = np.random.rand(N, K)
        P = np.double(P)
        Q = np.random.rand(M, K)
        Q = np.double(Q)

        ## 将兴趣归一矩阵利用LSH哈希函数将矩阵分块
        # final_matrix = lsh_duplicate_row_col_mc.row_mc(user_rank_matrix, user_style_matrix, P, Q, opts) # 对行进行计算
        final_matrix = lsh_duplicate_row_col_mc.lsh_mc(user_rank_matrix, user_style_matrix, P, Q, opts)  # 对行列进行分块

        # print(final_matrix)
        endtime_lsh = (datetime.datetime.now() - starttime).microseconds / 1000 / 60

        # 没有重叠只是重拍
        final_nodu_matrix = lsh_duplicate_row_col_mc.lsh_mc(user_rank_matrix, user_style_matrix, P, Q,
                                                            opts_no)  # 对行列进行分块
        # 直接用SGD方法
        starttime = datetime.datetime.now()
        direct_sgd_mc = sgd_test.sgd_test(user_rank_matrix, P, Q, opts)
        endtime_sgd = (datetime.datetime.now() - starttime).microseconds / 1000 / 60

        ## 随机分块方法
        starttime = datetime.datetime.now()
        rand_split_mc = rand_split_test.split_mc(user_rank_matrix, P, Q, opts)
        endtime_rand = (datetime.datetime.now() - starttime).microseconds / 1000 / 60
        # lmafit 方法
        # bool_idx = np.array(user_rank_matrix)
        # bool_idx[bool_idx>0] = 1
        # bool_idx = bool_idx.astype(int)
        # X,Y,Out = lmafit.lmafit_mc_adp(user_rank_matrix.shape[0],user_rank_matrix.shape[1],opts['rank'],bool_idx,user_rank_matrix,1)
        # lmafit_mc = X.dot(Y)
        ## 评价
        lsh_test_error = EVA.test_error(final_matrix, user_test_rank_matrix)
        lsh_test_nodu_error = EVA.test_error(final_nodu_matrix, user_test_rank_matrix)
        direct_test_error = EVA.test_error(direct_sgd_mc, user_test_rank_matrix)
        rand_test_error = EVA.test_error(rand_split_mc, user_test_rank_matrix)
        # lmafit_test_error = EVA.test_error(lmafit_mc,user_test_rank_matrix)
        # print("已经完成%d\n", rank)
        result_lsh_rank_matrix[i] = lsh_test_error
        result_sgd_rank_matrix[i] = direct_test_error
        result_lsh_time_matrix[i] = endtime_lsh
        result_sgd_time_matrix[i] = endtime_sgd
        result_rand_rank_matrix[i] = rand_test_error
        result_rand_time_matrix[i] = endtime_rand
        result_lsh_nodu_rank_matrix[i] = lsh_test_nodu_error
        i = i + 1
        print(rate, lsh_test_error, endtime_lsh, endtime_sgd)
        # print("sgd:%f\n" % direct_test_error)
        # print("lmafit:%f\n" % lmafit_test_error)
    result_rank = {
        'result_lsh_rank_matrix': result_lsh_rank_matrix,
        'result_sgd_rank_matrix': result_sgd_rank_matrix,
        'result_lsh_time_matrix': result_lsh_time_matrix,
        'result_sgd_time_matrix': result_sgd_time_matrix,
        'result_rand_rank_matrix': result_rand_rank_matrix,
        'result_rand_time_matrix': result_rand_time_matrix,
        'result_lsh_nodu_rank_matrix': result_lsh_nodu_rank_matrix
    }
    np.save('dataset/result/rank_lsh_pm25.npy', result_rank)
    # mat = np.load('dataset/result/rank_lsh_ping.npy')
    io.savemat('dataset/result/rank_lsh_pm25.mat', {'rank_lsh_pm25': result_rank})
