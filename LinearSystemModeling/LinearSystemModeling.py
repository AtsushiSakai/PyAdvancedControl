#! /usr/bin/python 
# -*- coding: utf-8 -*- 
u"""
Python library of Linear system modeling

author: Atsushi Sakai
"""
import numpy as np
import scipy.linalg as splinalg

def c2d(Ac,Bc,Ts,method="ZOH"):
    u"""
    Get system matrix of discrete system from continious system

    input:
        Ac:system matrix of continious system on dx=Ac x+Bc u
        Bc:system matrix of continious system on dx=Ac x+Bc u
        method: 
            - Euler: Euler discretization method
            - ZOH: Zero order hold method
    output:
        Ad: system matrix of discrete system xt+1=Ad xt + Bd u
        Bd: system matrix of discrete system xt+1=Ad xt + Bd u

        see: https://en.wikipedia.org/wiki/Discretization
    """

    if method=="Euler":
        # Euler Method
        A=np.eye(Ac.shape[0])+Ts*Ac
        B=Ts*Bc
    elif method=="ZOH": 
        #ZOH Discretization
        A=splinalg.expm(Ac*Ts)
        B=np.linalg.inv(Ac)*(A-np.eye(Ac.shape[0]))*Bc
    else:
        print("Error:Unknown method")
        print(method)

    return A,B


if __name__ == '__main__':
    Ac=np.matrix(np.zeros((4,4)))
    Ac[0,0]=-1.93
    Ac[1,0]=0.394
    Ac[1,1]=-0.426
    Ac[2,2]=-0.63
    Ac[3,0]=0.82
    Ac[3,1]=-0.784
    Ac[3,2]=0.413
    Ac[3,3]=-0.426
    print("Ac:")
    print(Ac)

    Bc=np.matrix(np.zeros((4,2)))
    Bc[0,0]=1.274
    Bc[0,1]=1.274
    Bc[2,0]=1.34
    Bc[2,1]=-0.65
    print("Bc:")
    print(Bc)

    Ts=2.0
    A,B=c2d(Ac,Bc,Ts)
    print("A")
    print(A)
    print("B")
    print(B)


