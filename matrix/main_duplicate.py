# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 01:10:35 2017

@author: yuxiang
"""
import numpy as np
import matrix.evaluate.Evaluate as EVA
import matrix.method.lsh_duplicate_mc as lsh_duplicate_mc
import matrix.test_dateset.ml100k as ml100k
import matrix.test_dateset.sgd_test as sgd_test
import matrix.util.io_file as io_file

if __name__ == "__main__":
    # 参数设定
    opts = {
        'k_number':10,                 # 哈希桶个数
        'w':1,                          # 桶宽
        'combin_number':2,              # 联合桶个数
        'split_number':2,               # 分块个数
        'b':np.random.uniform(0,1),         # 局部敏感哈希函数中b
        'rank':10,                      # 预估的矩阵的秩
        'alpha':0.0002,                 # sgd 参数
        'beta':0.02,                    # sgd 参数
        'step':1,                       # 循环计算次数
    }

    ## ml-100k 数据集
    # user_test_rank_matrix:    训练矩阵
    # user_rank_matrix:         原始矩阵
    # item_style_matrix:        电影类别矩阵
    # user_style_matrix:        用户电影类别矩阵(局部敏感哈希函数分类矩阵)
    file_url = 'dataset/ml-100k/ua.base'
    item_url = 'dataset/ml-100k/u.item'
    test_url = 'dataset/ml-100k/ua.test'
    user_test_rank_matrix, user_rank_matrix, item_style_matrix, user_style_matrix = ml100k.test_ml100k(file_url,item_url,test_url)

    ## geant 数据集
    file_url = 'dataset/gant/GeantOD.mat'
    save_url = 'dataset/gant'
    extra_name = 'GeantMatrixODNorm'
    io_file.save_mat(file_url,save_url,extra_name) # 将mat文件保存为npy文件
    save_url_name = save_url+"/"+extra_name+".npy"
    user_rank_matrix = np.load('save_url_name')

    ## 将兴趣归一矩阵利用LSH哈希函数将矩阵分块
    final_matrix = lsh_duplicate_mc.lsh_mc(user_rank_matrix,user_style_matrix,opts)
    # print(final_matrix)

    # 直接用SGD方法
    direct_sgd_mc = sgd_test.sgd_test(user_rank_matrix,opts)
    ## 评价
    lsh_test_error = EVA.test_error(final_matrix, user_test_rank_matrix)
    direct_test_error = EVA.test_error(direct_sgd_mc, user_test_rank_matrix)
    print("lsh:%f\n" % lsh_test_error)
    print("sgd:%f\n" % direct_test_error)
