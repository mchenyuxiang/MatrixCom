#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/24 20:06
# @Author  : mchenyuxiang
# @Email   : mchenyuxiang@foxmail.com
# @Site    : 
# @File    : cuda_sgd.py
# @Software: PyCharm
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import matrix.schedule.block_in_schedule as block_in_schedule

from pycuda.compiler import SourceModule

mod = SourceModule("""
#include <cstdio>
__global__ void matrix_sgd(float *a, float *b, float *c, int n, int m,int *schedule,int loop_number, int rank, int STEP, float alpha, float beta) 
{
   
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    printf("I am %d.%d.%d\\n", blockDim.y, blockIdx.y,threadIdx.y);
    int per_thread_in_block = blockDim.y;
    int block_num = blockIdx.y;
    int thread_num = threadIdx.y;
    int k;
    float sum = 0;
    int times = 0;
    
    while (times < STEP){
		times++;
		for (int row = 0; row < loop_number; row++){
			int row_number = schedule[row * n + col];			
			for (k = 0; k < rank; k++) {
			}
		}
	}

}
""")


def cuda_sgd(P, Q, ori_matrix, col_number, line_number, loc_line, loop_number, rank, STEP, alpha, beta):
    matrix_sgd = mod.get_function("matrix_sgd")

    # nP, nQ = SGD(R, P, Q, K)

    matrix_sgd(
        cuda.InOut(P), cuda.InOut(Q), cuda.In(ori_matrix), cuda.In(col_number), cuda.In(line_number), cuda.In(loc_line),
        cuda.In(loop_number), cuda.In(rank), cuda.In(STEP), cuda.In(alpha), cuda.In(beta),
        block=(1, 4, 1), grid=(1, 1)
    )

    return P, Q


if __name__ == "__main__":
    R = [
        [5, 3, 1, 1],
        [4, 4, 2, 1],
        [1, 1, 3, 5],
        [1, 4, 5, 4],
        [3, 1, 5, 4],
    ]

    R = np.array(R)

    N = len(R)
    M = len(R[0])
    rank = 2

    P = np.random.rand(N, rank).astype(np.float64)
    Q = np.random.rand(M, rank).astype(np.float64)

    line_temp, matrix = block_in_schedule.block_in_schedule(R)
    R = R.astype(np.float32)
    line = line_temp.astype(np.int)
    col_number  = int(R.shape[1])
    line_number = int(R.shape[0])
    loop_number = int(line.shape[0])
    STEP = 200
    alpha = 0.00055
    beta = 0.02
    P, Q = cuda_sgd(P, Q, R, np.uint32(col_number), np.uint32(line_number), line,np.uint32(loop_number), np.uint32(rank), np.uint32(STEP), np.float64(alpha), np.float64(beta))
    result = np.dot(P, Q.T)

    print(result)
# print(dest-a*b)
