#! /usr/bin/python 
# -*- coding: utf-8 -*- 
u""" 
Linear-Quadratic Regulator sample code

author Atsushi Sakai
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

simTime=3.0
dt=0.1

# x[k+1] = Ax[k] + Bu[k]
# y[k] = Cx[k]
A=np.matrix([[1.1,2.0],[0,0.95]])
B=np.matrix([0.0,0.0787]).T
C=np.matrix([-2,1])
Kopt=None

def Observation(x):
    y=C*x
    ry=float(y[0])
    return (ry)

def Process(x,u):
    x=A*x+B*u
    return (x)

def dlqr(A,B,Q,R):
    """Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    #ref Bertsekas, p.151
 
    #first, try to solve the ricatti equation
    X = np.matrix(la.solve_discrete_are(A, B, Q, R))
     
    #compute the LQR gain
    K = np.matrix(la.inv(B.T*X*B+R)*(B.T*X*A))
     
    eigVals, eigVecs = la.eig(A-B*K)
     
    return K, X, eigVals

def LQRController(x,u):
    global Kopt
    if Kopt is None:
        Kopt,X,ev=dlqr(A,B,C.T*np.eye(1)*C,np.eye(1))

    u=-Kopt*x
    return u

def Main():
    time=0.0
    u_history=[]
    y_history=[]
    time_history=[]

    x=np.matrix([3,1]).T
    u=np.matrix([0,0,0])

    while time<=simTime:
        u=LQRController(x,u)
        u0=float(u[0,0])
        x=Process(x,u0)
        y=Observation(x)

        u_history.append(u0)
        y_history.append(y)
        time_history.append(time)
        time+=dt

    plt.plot(time_history,u_history,"-r",label="input")
    plt.plot(time_history,y_history,"-b",label="output")
    plt.grid(True)
    plt.xlim([0,simTime])
    plt.legend()
    plt.show()

if __name__ == '__main__':
    Main()
