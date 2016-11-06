#! /usr/bin/python 
# -*- coding: utf-8 -*- 
u""" 
MPC Path Tracking Simulation

author Atsushi Sakai
"""

import cvxpy
import numpy as np
from cvxpy import *
import matplotlib.pyplot as plt
from math import * 
import time

def LinealizeCarModel(xb,u,dt,lr):
    u"""
    TODO conplete model
    """

    x=xb[0]
    y=xb[1]
    v=xb[2]
    theta=xb[3]

    a=u[0]
    beta=u[1]
    
    A=np.eye(xb.shape[0])
    A[0,2]=dt*cos(theta+beta)
    A[1,2]=dt*sin(theta+beta)
    A[3,2]=dt*sin(beta)/lr

    B=np.zeros((xb.shape[0],u.shape[0]))
    B[2,0]=dt
    B[3,1]=dt*v*cos(beta)/lr

    return A,B

def CalcInput(A,B,x,u):

    T = 10  #number of horizon

    ustar=[]
    x_0 =x

    x = Variable(x.shape[0], T+1)
    u = Variable(u.shape[0], T)

    #MPC controller
    states = []
    for t in range(T):
        cost = sum_squares(u[:,t])
        #  cost = sum_squares(0)
        if t == T-1:
            #  cost+=sum_squares(x[1,t]-10.0)*1000
            cost+=sum_squares(x[0,t]-10.0)*1000
            cost+=sum_squares(x[1,t]-10.0)*1000
        constr = [x[:,t+1] == A*x[:,t] + B*u[:,t]]
        states.append( Problem(Minimize(cost), constr) )

    prob = sum(states)
    prob.constraints += [x[:,0] == x_0]

    start = time.time()
    #  result=prob.solve(verbose=True)
    result=prob.solve()
    elapsed_time = time.time() - start
    print ("calc time:{0}".format(elapsed_time)) + "[sec]"

    ustar=u

    #  print(x.value)
    print(u.value)

    return ustar


def Main():

    x0=np.matrix([0.0,0.0,0.0,0.0]).T#[x,y,v theta]
    x=x0
    u=np.matrix([0.0,0.00]).T#[a,beta]
    dt=0.1#[s]
    lr=1.0#[m]

    for i in range(1000):
        A,B=LinealizeCarModel(x,u,dt,lr)
        ustar=CalcInput(A,B,x,u)
        
        u[0,0]=float(ustar[0,0].value)
        u[1,0]=float(ustar[1,0].value)

        x=A*x+B*u

        plt.plot(x[0],x[1],'.r')
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)


if __name__ == '__main__':
    Main()
