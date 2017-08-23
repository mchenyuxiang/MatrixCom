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
import codecs
import datetime
import time

def save_mat(file_url,save_url,extra_name):
    mat = io.loadmat(file_url)
    mat_t = mat[extra_name]
    save_name = save_url+"/"+extra_name+".npy"
    np.save(save_name,mat_t)

def handle_airquality(ori_file_url,save_url,extra_name):
    air_matrix = np.zeros((437,8760))
    f = codecs.open(file_url,'r',encoding="utf8")
    # f = csv.reader(csvfile)
    i = -1
    j = 0
    last_line = 1
    line_cnt = 0
    d1 = time.mktime(datetime.datetime(2014,5,1,0,0).timetuple())
    for line in f:
        line_cnt = line_cnt + 1
        # line = line.strip("\r\n")
        # line = line.decode("utf-8")
        temp_line = line.split(",")

        d2_temp = datetime.datetime.strptime(temp_line[1],'%Y-%m-%d %H:%M:%S')
        d2 = time.mktime(d2_temp.timetuple())
        j = int((d2-d1)/3600)
        print(line_cnt,i,j)
        if int(temp_line[0])!= last_line:
            last_line = int(temp_line[0])
            i = i+1
            if i == 437:
                break
        air_matrix[i,j] = temp_line[2]
        # print(line.row)
        # print(line_cnt)
    save_name = save_url+"/"+extra_name+".npy"
    np.save(save_name, air_matrix)
    io.savemat(save_url+"/"+extra_name+".mat", {'air_matrix': air_matrix})

if __name__ == "__main__":
    # file_url = '../dataset/weather/PM25.mat'
    # save_url = '../dataset/weather'
    # extra_name = 'PM_Data'
    # save_mat(file_url,save_url,extra_name)

    file_url = '../dataset/airquality/airquality.txt'
    save_url = '../dataset/airquality'
    extra_name = 'airquality'
    handle_airquality(file_url,save_url,extra_name)