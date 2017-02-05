#! /usr/bin/python
# -*- coding: utf-8 -*

import cvxpy
import numpy as np
import matplotlib.pyplot as plt

import cvxopt
from cvxopt import matrix
import scipy.linalg


def use_modeling_tool(A, B, N, Q, R, P, umax, x0):
    (nx, nu) = B.shape

    # mpc calculation
    x = cvxpy.Variable(nx, N + 1)
    u = cvxpy.Variable(nu, N)

    costlist = 0.0
    constrlist = []

    for t in range(N):
        costlist += 0.5 * cvxpy.quad_form(x[:, t], Q)
        costlist += 0.5 * cvxpy.quad_form(u[:, t], R)

        constrlist += [x[:, t + 1] == A * x[:, t] + B * u[:, t]]

    costlist += 0.5 * cvxpy.quad_form(x[:, N], P)  # terminal cost

    prob = cvxpy.Problem(cvxpy.Minimize(costlist), constrlist)

    prob.constraints += [x[:, 0] == x0]  # inital state constraints
    #  prob.constraints += [cvxpy.abs(u) <= umax]  # input constraints

    prob.solve(verbose=True)

    rx1 = np.array(x.value[0, :]).flatten()
    rx2 = np.array(x.value[1, :]).flatten()
    ru = np.array(u.value[0, :]).flatten()

    flg, ax = plt.subplots(1)
    plt.plot(rx1, label="x1")
    plt.plot(rx2, label="x2")
    plt.plot(ru, label="u")
    plt.legend()
    plt.grid(True)
    #  plt.show()


def hand_modeling(A, B, N, Q, R, P, umax, x0):

    # calc AA
    Ai = A
    AA = Ai
    for i in range(2, N + 1):
        Ai = A * Ai
        AA = np.concatenate((AA, Ai), axis=0)
    #  print(AA)

    # calc BB
    AiB = B
    BB = np.kron(np.eye(N), AiB)
    for i in range(1, N):
        AiB = A * AiB
        BB += np.kron(np.diag(np.ones(N - i), -i), AiB)
    #  print(BB)

    RR = np.kron(np.eye(N), R)
    QQ = scipy.linalg.block_diag(np.kron(np.eye(N - 1), Q), P)

    H = (BB.T * QQ * BB + RR)
    #  print(H)

    gx0 = BB.T * QQ * AA * x0
    #  print(gx0)

    P = matrix(H)
    q = matrix(gx0)
    sol = cvxopt.solvers.qp(P, q)
    #  print(sol)

    u = np.matrix(sol["x"])

    # recover x
    xx = AA * x0 + BB * u
    x = np.concatenate((x0.T, xx.reshape(N, 2)), axis=0)

    x1 = np.array(x[:, 0]).flatten()
    x2 = np.array(x[:, 1]).flatten()

    #  flg, ax = plt.subplots(1)
    plt.plot(x1, '*r', label="x1")
    plt.plot(x2, '*b', label="x2")
    plt.plot(u, '*k', label="u")
    plt.legend()
    plt.grid(True)


def main():
    print("start!!")
    A = np.matrix([[0.8, 1.0], [0, 0.9]])
    print(A)
    B = np.matrix([[-1.0], [2.0]])
    print(B)
    (nx, nu) = B.shape
    print(nx, nu)

    N = 10  # number of horizon
    Q = np.eye(nx)
    print(Q)
    R = np.eye(nu)
    print(R)
    P = np.eye(nx)
    print(P)
    umax = 0.7

    x0 = np.matrix([[1.0], [2.0]])  # init state

    use_modeling_tool(A, B, N, Q, R, P, umax, x0)
    hand_modeling(A, B, N, Q, R, P, umax, x0)

    plt.show()


if __name__ == '__main__':
    main()
