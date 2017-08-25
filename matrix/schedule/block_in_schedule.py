#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/24 13:53
# @Author  : mchenyuxiang
# @Email   : mchenyuxiang@foxmail.com
# @Site    : 
# @File    : block_in_schedule.py
# @Software: PyCharm
import numpy as np

## block内部调度策略
def block_in_schedule(ori_matrix):
    sample_matrix = np.array(ori_matrix)
    col_number = ori_matrix.shape[1]
    sample_matrix[sample_matrix>0]=1 # 得到采样位置
    loop_number = np.max(np.sum(sample_matrix,axis=0))  # 最大的循环次数
    loc_line = np.zeros((loop_number,col_number))   #   每一次循环的行
    loc_matrix = np.zeros((loop_number,col_number))
    for i in range(loop_number):
        flag_line = np.zeros(ori_matrix.shape[0])   # 行是否被使用的fla
        for j in range(col_number):
            sample_number = np.sum(sample_matrix[:,j])
            if sample_number == 0:
                sample_matrix[:,j] = ori_matrix[:,j]
            temp_col = sample_matrix[:,j]*sample_matrix[:,j]
            no_zero_index = np.where(temp_col>0)[0]    #    找出不为0的位置
            np.random.shuffle(no_zero_index)    #   随机打乱排序
            for k in range(len(no_zero_index)):
                loc_line_temp =  no_zero_index[k]
                if flag_line[loc_line_temp] != 1:
                    loc_line[i,j] = loc_line_temp
                    flag_line[loc_line_temp] = 1
                    loc_matrix[i,j] = ori_matrix[loc_line_temp,j]
                    sample_matrix[loc_line_temp,j] = 0
                    break

    return loc_line,loc_matrix

if __name__ == "__main__":
    a = np.array([(1,2,0,4),(2,3,0,5),(0,5,6,7),(6,7,8,9)])
    line,matrix = block_in_schedule(a)
    print(line)
    print(matrix)

