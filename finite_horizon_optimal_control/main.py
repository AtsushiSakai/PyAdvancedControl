#! /usr/bin/python 
# -*- coding: utf-8 -*- 
u""" 
Finite Horizon Optimal Control

author Atsushi Sakai
"""

import numpy as np
import scipy.linalg as la

def CalcFiniteHorizonOptimalInput(A,B,Q,R,P,N,x0):
    u"""
    Calc Finite Horizon Optimal Input

    # TODO optimize
        size check
        search kron
    """
    print("CalcFiniteHorizonOptimalInput start")

    sx=np.eye(A.ndim)
    su=np.zeros((A.ndim,B.shape[1]*N))

    #calc sx,su
    for i in range(N):
        #generate sx
        An=np.linalg.matrix_power(A, i+1)
        sx=np.r_[sx,An]

        #generate su
        tmp=None
        for ii in range(i+1):
            tm=np.linalg.matrix_power(A, ii)*B
            if tmp is None: 
                tmp=tm
            else:
                tmp =np.c_[tm,tmp]

        for ii in np.arange(i,N-1):
            tm=np.zeros(B.shape)
            if tmp is None: 
                tmp=tm
            else:
                tmp =np.c_[tmp,tm]

        su=np.r_[su,tmp]

    tm1=np.eye(N+1)
    tm1[N,N]=0
    tm2=np.zeros((N+1,N+1))
    tm2[N,N]=1
    Qbar=np.kron(tm1,Q)+np.kron(tm2,P)
    Rbar=np.kron(np.eye(N),R)

    uBa=-(su.T*Qbar*su+Rbar).I*su.T*Qbar*sx*x0
    #  print(uBa)
    costBa=x0.T*(sx.T*Qbar*sx-sx.T*Qbar*su*(su.T*Qbar*su+Rbar).I*su.T*Qbar*sx)*x0
    #  print(costBa)

    #  print(su)
    #  print(sx)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    A=np.matrix([[0.77,-0.35],[0.49,0.91]])
    print("A:")
    print(A)
    B=np.matrix([0.04,0.15]).T
    print("B:")
    print(B)
    x0=np.matrix([1,-1]).T
    print("x0")
    print(x0)
    Q=np.matrix([[500,0.0],[0.0,100]])
    print("Q")
    print(Q)
    R=np.matrix([1.0])
    print("R")
    print(R)
    P=np.matrix([[1500,0.0],[0.0,100]])
    print("P")
    print(P)
    N=3#Number of horizon

    ustar=CalcFiniteHorizonOptimalInput(A,B,Q,R,P,N,x0)


