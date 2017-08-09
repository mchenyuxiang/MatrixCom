#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/8 17:36
# @Author  : mchenyuxiang
# @Email   : mchenyuxiang@foxmail.com
# @Site    : 
# @File    : HandleFile.py
# @Software: PyCharm

import numpy as np
import codecs

## 创建评分矩阵
def create_matrix(file_url,line_size,col_size,split_symbol):
    rate_matrix = np.zeros((line_size,col_size))
    f = open(file_url)
    for line in f:
        line = line.strip('\n')
        temp_line = line.split(split_symbol)
        rate_matrix[int(temp_line[0])-1,int(temp_line[1])-1] = float(temp_line[2])
    # print(rate_matrix[0,0])
    return rate_matrix

## 创建类别矩阵
def create_file_style_matrix(file_url,line_size,col_size,split_symbol):
    rate_matrix = np.zeros((line_size,col_size))
    f = codecs.open(file_url,'r','latin-1')
    for line in f:
        line = line.strip('\n')
        temp_line = line.split(split_symbol)
        attr_list = temp_line[5:24]
        for index,item in enumerate(attr_list):
            rate_matrix[int(temp_line[0])-1,index] = int(item)
    # print(rate_matrix[0])
        # rate_matrix[int(temp_line[0])-1,int(temp_line[1])-1] = int(temp_line[2])
    return rate_matrix

## 用户电影兴趣归一矩阵
def create_user_style_matrix(file_url,item_matrix,line_size,col_size,split_symbol):
    rate_matrix = np.zeros((line_size,col_size))
    number_matrix = np.zeros((line_size,col_size))
    f = open(file_url)
    for line in f:
        line = line.strip('\n')
        temp_line = line.split(split_symbol)
        rate_matrix[int(temp_line[0])-1] = rate_matrix[int(temp_line[0])-1] + item_matrix[int(temp_line[1])-1]*float(temp_line[2])
        number_matrix[int(temp_line[0])-1] = number_matrix[int(temp_line[0])-1] + item_matrix[int(temp_line[1])-1]
    number_matrix[number_matrix==0]=1
    rate_matrix = rate_matrix / number_matrix
    return rate_matrix
    # print(rate_matrix[0])

if __name__ == "__main__":
    file_url = '../dataset/ml-100k/ua.base'
    item_url = '../dataset/ml-100k/u.item'
    user_size = 943
    item_size = 1682
    style_size = 19
    # user_rank_matrix = create_matrix(file_url,user_size,item_size,"\t")
    item_style_matrix = create_file_style_matrix(item_url,item_size,style_size,"|")
    create_user_style_matrix(file_url,item_style_matrix,user_size,style_size,"\t")
