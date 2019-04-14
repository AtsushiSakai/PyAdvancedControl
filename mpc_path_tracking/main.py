""" 
MPC driving simulation to target point 

author Atsushi Sakai
"""

import cvxpy
import numpy as np
from cvxpy import *
import matplotlib.pyplot as plt
from math import *
import time

dt = 0.1  # [s] discrete time
lr = 1.0  # [m]
T = 15  # number of horizon
target = [5.0, -1.0]  # [x,y]

max_speed = 5.0
min_speed = -5.0


def LinealizeCarModel(xb, u, dt, lr):
    """
    TODO conplete model
    """

    x = xb[0]
    y = xb[1]
    v = xb[2]
    theta = xb[3]

    a = u[0]
    beta = u[1]

    t1 = -dt * v * sin(theta + beta)
    t2 = dt * v * cos(theta + beta)

    A = np.eye(xb.shape[0])
    A[0, 2] = dt * cos(theta + beta)
    A[1, 2] = dt * sin(theta + beta)
    A[3, 2] = dt * sin(beta) / lr
    A[0, 3] = t1
    A[1, 3] = t2

    B = np.zeros((xb.shape[0], u.shape[0]))
    B[2, 0] = dt
    B[0, 1] = t1
    B[1, 1] = t2
    B[3, 1] = dt * v * cos(beta) / lr

    tm = np.zeros((4, 1))
    tm[0, 0] = v * cos(theta + beta) * dt
    tm[1, 0] = v * sin(theta + beta) * dt
    tm[2, 0] = a * dt
    tm[3, 0] = v / lr * sin(beta) * dt
    C = xb + tm
    C = C - A @ xb - B @ u

    # print(A, B, C)

    return A, B, C


def NonlinearModel(x, u, dt, lr):
    print(x.value)
    x[0] = x[0] + x[2] * cos(x[3] + u[1]) * dt
    x[1] = x[1] + x[2] * sin(x[3] + u[1]) * dt
    x[2] = x[2] + u[0] * dt
    x[3] = x[3] + x[2] / lr * sin(u[1]) * dt

    return x


def CalcInput(A, B, C, x, u):

    x_0 = x[:]
    x = Variable((x.shape[0], T + 1))
    u = Variable((u.shape[0], T))

    # MPC controller
    states = []
    for t in range(T):
        constr = [x[:, t + 1] == A * x[:, t] + B * u[:, t] + C]
        constr += [abs(u[:, t]) <= 0.5]
        constr += [x[2, t + 1] <= max_speed]
        constr += [x[2, t + 1] >= min_speed]
        #  cost = sum_squares(u[:,t])
        cost = sum_squares(abs(x[0, t] - target[0])) * 10.0 * t
        cost += sum_squares(abs(x[1, t] - target[1])) * 10.0 * t
        if t == T - 1:
            cost += (x[0, t + 1] - target[0]) ** 2 * 10000.0
            cost += (x[1, t + 1] - target[1]) ** 2 * 10000.0

        states.append(Problem(Minimize(cost), constr))

    prob = sum(states)
    prob.constraints += [x[:, 0] == x_0, x[2, T] == 0.0]

    start = time.time()
    #  result=prob.solve(verbose=True)
    result = prob.solve()
    elapsed_time = time.time() - start
    print("calc time:{0}".format(elapsed_time) + "[sec]")
    print(prob.value)

    if prob.status != OPTIMAL:
        print("Cannot calc opt")

    #  print(prob.status)
    return u, x, prob.value


def GetListFromMatrix(x):
    return np.array(x).flatten().tolist()


def Main():
    x0 = np.array([[0.0, 0.0, 0.0, 0.0]]).T  # [x,y,v theta]
    print(x0)
    x = x0
    u = np.array([[0.0, 0.0]]).T  # [a,beta]
    plt.figure(num=None, figsize=(12, 12))

    mincost = 100000

    for i in range(1000):
        A, B, C = LinealizeCarModel(x, u, dt, lr)
        ustar, xstar, cost = CalcInput(A, B, C, x, u)

        u[0, 0] = GetListFromMatrix(ustar.value[0, :])[0]
        u[1, 0] = float(ustar[1, 0].value)

        x = A @ x + B @ u

        plt.subplot(3, 1, 1)
        plt.plot(target[0], target[1], 'xb')
        plt.plot(x[0], x[1], '.r')
        plt.plot(GetListFromMatrix(xstar.value[0, :]), GetListFromMatrix(
            xstar.value[1, :]), '-b')
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.cla()
        plt.plot(GetListFromMatrix(xstar.value[2, :]), '-b')
        plt.plot(GetListFromMatrix(xstar.value[3, :]), '-r')
        plt.ylim([-1.0, 1.0])
        plt.ylabel("velocity[m/s]")
        plt.xlabel("horizon")
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.cla()
        plt.plot(GetListFromMatrix(ustar.value[0, :]), '-r', label="a")
        plt.plot(GetListFromMatrix(ustar.value[1, :]), '-b', label="b")
        plt.ylim([-0.5, 0.5])
        plt.legend()
        plt.grid(True)

        #  plt.pause(0.0001)

        #  raw_input()

        # check goal
        dis = np.linalg.norm([x[0] - target[0], x[1] - target[1]])
        if (dis < 0.1):
            print("Goal")
            break

    plt.show()


if __name__ == '__main__':
    Main()
