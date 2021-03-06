# -*- coding: utf-8 -*-

import numpy
from numba import jit

@jit
def SGD(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02,tol=1e-7):
    Q = Q.T
    P = numpy.float64(P)
    Q = numpy.float64(Q)
    # e_old = 10000000
    numpy.seterr(all='print')
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        # eR = numpy.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        # if step == 0:
        #     e_old = e + 1
        if e < tol:
            break
        # print(e)
        # if numpy.abs(e-e_old) < tol:
        #     break
        # if (e-e_old) > 0:
        #     break
        # e_old = e
    return P, Q.T

@jit
def SGD_weight(R, P, Q, K,distance_matrix, steps=5000, alpha=0.0002, beta=0.02,tol=1e-7):
    Q = Q.T
    P = numpy.float64(P)
    Q = numpy.float64(Q)
    # e_old = 10000000
    numpy.seterr(all='print')
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    eij = distance_matrix[i,j] * eij
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        # eR = numpy.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        # if step == 0:
        #     e_old = e + 1
        if e < tol:
            break
        # print(e)
        # if numpy.abs(e-e_old) < tol:
        #     break
        # if (e-e_old) > 0:
        #     break
        # e_old = e
    return P, Q.T

if __name__ == "__main__":
    # R = [
    #      [5,3,0,1],
    #      [4,0,0,1],
    #      [1,1,0,5],
    #      [1,0,0,4],
    #      [0,1,5,4],
    #     ]
    R = [
        [5, 3, 0, 1]
    ]

    R = numpy.array(R)

    N = len(R)
    M = len(R[0])
    K = 2

    P = numpy.random.rand(N,K)
    Q = numpy.random.rand(M,K)

    nP, nQ = SGD(R, P, Q, K)
    
    result = numpy.dot(nP,nQ.T)

    print(result)