#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/12 17:53
# @Author  : mchenyuxiang
# @Email   : mchenyuxiang@foxmail.com
# @Site    : 
# @File    : ml100k.py
# @Software: PyCharm

import matrix.util.HandleFile as HF

def test_ml100k(file_url,item_url,test_url):
    user_size = 943
    item_size = 1682
    style_size = 19

    ## 建立用户兴趣归一矩阵
    user_test_rank_matrix = HF.create_matrix(test_url, user_size, item_size, "\t")
    user_rank_matrix = HF.create_matrix(file_url, user_size, item_size, "\t")
    item_style_matrix = HF.create_file_style_matrix(item_url, item_size, style_size, "|")
    user_style_matrix = HF.create_user_style_matrix(file_url, item_style_matrix, user_size, style_size, "\t")
    return user_test_rank_matrix,user_rank_matrix,item_style_matrix,user_style_matrix