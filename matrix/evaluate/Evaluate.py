#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/9 10:13
# @Author  : mchenyuxiang
# @Email   : mchenyuxiang@foxmail.com
# @Site    : 
# @File    : Evaluate.py
# @Software: PyCharm

import numpy as np
import numpy.linalg as LA

## 测试数据恢复误差
def test_error(computer_matrix,test_matrix):
    idx = np.where(test_matrix > 0)
    test_location = np.zeros((test_matrix.shape[0],test_matrix.shape[1]))
    test_location[idx] = 1
    # print(test_location)
    compare_matrix = computer_matrix * test_location
    test_error_f = LA.norm((compare_matrix-test_matrix),'fro')
    return test_error_f
    pass


if __name__ == "__main__":
    test_matrix = np.array([(2,0,3,0),(0,4,0,4)])
    computer_matrix = np.array([(3,0,3,0),(0,4,0,4)])
    fro = test_error(computer_matrix,test_matrix)
    print(fro)
    pass