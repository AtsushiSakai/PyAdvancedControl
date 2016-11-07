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

dt=0.5  #[s] discrete time
lr=1.0  #[m]
T = 10  #number of horizon
target=[-10,5]#[x,y]

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

    t1=-dt*v*sin(theta+beta)
    t2=dt*v*cos(theta+beta)
    
    A=np.eye(xb.shape[0])
    A[0,2]=dt*cos(theta+beta)
    A[1,2]=dt*sin(theta+beta)
    A[3,2]=dt*sin(beta)/lr
    A[0,3]=t1
    A[1,3]=t2

    B=np.zeros((xb.shape[0],u.shape[0]))
    B[2,0]=dt
    B[0,1]=t1
    B[1,1]=t2
    B[3,1]=dt*v*cos(beta)/lr

    return A,B

def CalcInput(A,B,x,u):

    x_0 =x
    x = Variable(x.shape[0], T+1)
    u = Variable(u.shape[0], T)

    #MPC controller
    states = []
    for t in range(T):
        cost = sum_squares(u[:,t])
        cost+=sum_squares(x[0,t]-target[0])*10.0
        cost+=sum_squares(x[1,t]-target[1])*10.0
        if t==T-1:
            cost+=sum_squares(x[0,t]-target[0])*1000.0
            cost+=sum_squares(x[1,t]-target[1])*1000.0
 
        constr = [x[:,t+1] == A*x[:,t] + B*u[:,t], abs(u[:,t])<=0.1,abs(x[2,t])<= 5.0]
        states.append( Problem(Minimize(cost), constr) )

    prob = sum(states)
    prob.constraints += [x[:,0] == x_0,x[2,T] == 0.0]

    start = time.time()
    #  result=prob.solve(verbose=True)
    result=prob.solve()
    elapsed_time = time.time() - start
    print ("calc time:{0}".format(elapsed_time)) + "[sec]"

    #  print(prob.status)
    return u,x

def GetListFromMatrix(x):
    return np.array(x).flatten().tolist()


def Main():
    x0=np.matrix([0.0,0.0,0.0,0.0]).T#[x,y,v theta]
    x=x0
    u=np.matrix([0.0,0.00]).T#[a,beta]

    for i in range(100):
        A,B=LinealizeCarModel(x,u,dt,lr)
        ustar,xstar=CalcInput(A,B,x,u)
        
        u[0,0]=GetListFromMatrix(ustar.value[0,:])[0]
        u[1,0]=float(ustar[1,0].value)

        x=A*x+B*u

        plt.subplot(2, 1, 1)
        plt.plot(target[0],target[1],'xb')
        plt.plot(x[0],x[1],'.r')
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.cla()
        plt.plot(GetListFromMatrix(xstar.value[2,:]),'-b')
        plt.ylim([-1.0,1.0])
        plt.ylabel("velocity[m/s]")
        plt.xlabel("horizon")

        plt.grid(True)
        plt.pause(0.0001)

        dis=np.linalg.norm([x[0]-target[0],x[1]-target[1]])
        if (dis<0.1):
            print("Goal")
            break

    plt.show()


if __name__ == '__main__':
    Main()
