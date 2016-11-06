#! /usr/bin/python 
# -*- coding: utf-8 -*- 
u""" 
Simple Model Predictive Control Simulation

author Atsushi Sakai
"""
import time
from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt
print("Simulation start")

np.random.seed(1)
n = 4   #state size
m = 2   #input size
T = 50  #number of horizon

#simulation parameter
alpha = 0.2
beta = 5.0

# Model Parameter
A = np.eye(n) + alpha*np.random.randn(n,n)
B = np.random.randn(n,m)
x_0 = beta*np.random.randn(n,1)

x = Variable(n, T+1)
u = Variable(m, T)

states = []
for t in range(T):
    cost = sum_squares(x[:,t+1]) + sum_squares(u[:,t])
    constr = [x[:,t+1] == A*x[:,t] + B*u[:,t],
              norm(u[:,t], 'inf') <= 1]
    states.append( Problem(Minimize(cost), constr) )
# sums problem objectives and concatenates constraints.
prob = sum(states)
prob.constraints += [x[:,T] == 0, x[:,0] == x_0]

start = time.time()
result=prob.solve(verbose=True)
elapsed_time = time.time() - start
print ("calc time:{0}".format(elapsed_time)) + "[sec]"

if result == float("inf"):
    print("Cannot optimize")
    import sys
    sys.exit()
    #  return

f = plt.figure()
# Plot (u_t)_1.
ax = f.add_subplot(211)
u1=np.array(u[0,:].value[0,:])[0].tolist()
u2=np.array(u[1,:].value[0,:])[0].tolist()
plt.plot(u1,'-r',label="u1")
plt.plot(u2,'-b',label="u2")
plt.ylabel(r"$u_t$", fontsize=16)
plt.yticks(np.linspace(-1.0, 1.0, 3))
plt.legend()
plt.grid(True)

# Plot (u_t)_2.
plt.subplot(2,1,2)
x1=np.array(x[0,:].value[0,:])[0].tolist()
x2=np.array(x[1,:].value[0,:])[0].tolist()
x3=np.array(x[2,:].value[0,:])[0].tolist()
x4=np.array(x[3,:].value[0,:])[0].tolist()
plt.plot(range(T+1), x1,'-r',label="x1")
plt.plot(range(T+1), x2,'-b',label="x2")
plt.plot(range(T+1), x3,'-g',label="x3")
plt.plot(range(T+1), x4,'-k',label="x4")
plt.yticks([-25, 0, 25])
plt.ylim([-25, 25])
plt.ylabel(r"$x_t$", fontsize=16)
plt.xlabel(r"$t$", fontsize=16)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
