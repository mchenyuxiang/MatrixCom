# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 01:10:35 2017

@author: yuxiang
"""
import numpy as np
import matrix.evaluate.Evaluate as EVA
import matrix.method.lsh_duplicate_row_col_mc as lsh_duplicate_row_col_mc
import matrix.test_dateset.sgd_test as sgd_test
import matrix.sample.line_sample as line_sample
import matrix.mc.lmafit as lmafit

if __name__ == "__main__":
    # 参数设定
    k_number = 10
    w = 0.0001
    combin_number = 2
    split_number = 50
    split_col_number = 1
    b = np.random.uniform(0, w)
    rank = 8
    aplha = 0.00055
    beta = 0.02
    step_lsh = 400
    step_sgd = 1000
    rate = 0.5
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
    save_url = 'dataset/pings_4hr'
    extra_name = 'originMatrix4h30min'
    save_url_name = save_url + "/" + extra_name + ".npy"
    user_ori_matrix = np.load(save_url_name)

    # extra_name = 'dataset/result/rank_alpha_lsh'
    # extra_name = 'GeantMatrixOD'
    # io_file.save_mat(file_url,save_url,extra_name) # 将mat文件保存为npy文件
    # save_url_name = extra_name + ".npy"
    flag = 1
    while flag==1:
        for i in range(len(user_ori_matrix)):
            if np.sum(user_ori_matrix[i]) == 0:
                user_ori_matrix = np.delete(user_ori_matrix,i,0)
                break
        if i == len(user_ori_matrix)-1:
            flag = 0

    user_ori_matrix = user_ori_matrix/1000
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

    # print(b)
    user_rank_matrix = line_sample.line_sample_matrix(user_ori_matrix, user_sample_squence, opts['rate'])  # 生成训练矩阵
    print("================end sample==============")
    user_style_matrix = user_rank_matrix
    user_test_rank_matrix = user_ori_matrix - user_rank_matrix  # 生成验证结果矩阵
    # print(c)


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

    # 直接用SGD方法
    direct_sgd_mc = sgd_test.sgd_test(user_rank_matrix, P, Q, opts)

    # lmafit 方法
    # bool_idx = np.array(user_rank_matrix)
    # bool_idx[bool_idx>0] = 1
    # bool_idx = bool_idx.astype(int)
    # X,Y,Out = lmafit.lmafit_mc_adp(user_rank_matrix.shape[0],user_rank_matrix.shape[1],opts['rank'],bool_idx,user_rank_matrix,1)
    # lmafit_mc = X.dot(Y)
    ## 评价
    lsh_test_error = EVA.test_error(final_matrix, user_test_rank_matrix)
    direct_test_error = EVA.test_error(direct_sgd_mc, user_test_rank_matrix)
    # lmafit_test_error = EVA.test_error(lmafit_mc,user_test_rank_matrix)
    # print("已经完成%d\n", rank)
    print("lsh:%f\n" % lsh_test_error)
    print("sgd:%f\n" % direct_test_error)
    # print("lmafit:%f\n" % lmafit_test_error)
