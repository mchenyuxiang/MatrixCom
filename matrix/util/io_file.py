#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/12 18:18
# @Author  : mchenyuxiang
# @Email   : mchenyuxiang@foxmail.com
# @Site    : 
# @File    : io_file.py
# @Software: PyCharm

import numpy as np
from scipy import io

def save_mat(file_url,save_url,extra_name):
    mat = io.loadmat(file_url)
    mat_t = mat[extra_name]
    save_name = save_url+"/"+extra_name+".npy"
    np.save(save_name,mat_t)

if __name__ == "__main__":
    file_url = '../dataset/pings_4hr/pings_4h30min.mat'
    save_url = '../dataset/pings_4hr'
    extra_name = 'originMatrix4h30min'
    save_mat(file_url,save_url,extra_name)