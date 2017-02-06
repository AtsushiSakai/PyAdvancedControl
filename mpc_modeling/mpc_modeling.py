#! /usr/bin/python
# -*- coding: utf-8 -*

import cvxpy
import numpy as np
import matplotlib.pyplot as plt

import cvxopt
from cvxopt import matrix
import scipy.linalg


def use_modeling_tool(A, B, N, Q, R, P, x0, umax=None, umin=None, xmin=None, xmax=None):
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

        if xmin is not None:
            constrlist += [x[:, t] >= xmin]
        if xmax is not None:
            constrlist += [x[:, t] <= xmax]

    costlist += 0.5 * cvxpy.quad_form(x[:, N], P)  # terminal cost
    if xmin is not None:
        constrlist += [x[:, N] >= xmin]
    if xmax is not None:
        constrlist += [x[:, N] <= xmax]

    prob = cvxpy.Problem(cvxpy.Minimize(costlist), constrlist)

    prob.constraints += [x[:, 0] == x0]  # inital state constraints
    if umax is not None:
        prob.constraints += [u <= umax]  # input constraints
    if umin is not None:
        prob.constraints += [u >= umin]  # input constraints

    prob.solve(verbose=True)

    return x.value, u.value


def hand_modeling(A, B, N, Q, R, P, x0, umax=None, umin=None):
    (nx, nu) = B.shape

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

    if umax is None and umin is None:
        P = matrix(H)
        q = matrix(gx0)
        sol = cvxopt.solvers.qp(P, q)
        #  print(sol)
    else:
        P = matrix(H)
        q = matrix(gx0)

        G = np.matrix([])
        h = np.matrix([])

        if umax is not None:
            G = np.eye(N)
            h = np.ones((N, 1)) * umax

        if umin is not None:
            if umax is None:
                G = np.eye(N) * -1.0
                h = np.ones((N, 1)) * umin * -1.0
            else:
                G = np.concatenate((G, np.eye(N) * -1.0), axis=0)
                h = np.concatenate((h, np.ones((N, 1)) * umin * -1.0), axis=0)

        G = matrix(G)
        h = matrix(h)

        sol = cvxopt.solvers.qp(P, q, G, h)

    u = np.matrix(sol["x"])

    # recover x
    xx = AA * x0 + BB * u
    x = np.concatenate((x0.T, xx.reshape(N, nx)), axis=0)

    return x, u


def hand_modeling2(A, B, N, Q, R, P, x0, xmin, xmax, umax=None, umin=None):
    (nx, nu) = B.shape

    H = scipy.linalg.block_diag(np.kron(np.eye(N), R), np.kron(np.eye(N - 1), Q), np.eye(P.shape[0]))
    #  print(H)

    # calc Ae
    Aeu = np.kron(np.eye(N), -B)
    #  print(Aeu)
    #  print(Aeu.shape)
    Aex = scipy.linalg.block_diag(np.eye((N - 1) * nx), P)
    Aex -= np.kron(np.diag([1.0] * (N - 1), k=-1), A)
    #  print(Aex)
    #  print(Aex.shape)
    Ae = np.concatenate((Aeu, Aex), axis=1)
    #  print(Ae.shape)

    # calc be
    #  be = [Azeros((N - 1) * nx, nx)] * x0
    be = np.concatenate((A, np.zeros(((N - 1) * nx, nx))), axis=0) * x0
    #  print(be)

    P = matrix(H)
    q = matrix(np.zeros((N * nx + N * nu, 1)))
    A = matrix(Ae)
    b = matrix(be)

    sol = cvxopt.solvers.qp(P, q, A=A, b=b)
    #  print(sol)
    fx = np.matrix(sol["x"])
    #  print(fx)

    u = fx[0:N * nu].reshape(N, nu).T
    x = fx[-N * nx:].reshape(N, nx).T
    x = np.concatenate((x0, x), axis=1)
    #  print(x)
    #  print(u)

    return x, u


def test2():
    print("start!!")
    A = np.matrix([[0.8, 1.0], [0, 0.9]])
    B = np.matrix([[-1.0], [2.0]])
    (nx, nu) = B.shape

    N = 10  # number of horizon
    Q = np.eye(nx)
    R = np.eye(nu)
    P = np.eye(nx)
    umax = 0.7
    umin = -0.7

    x0 = np.matrix([[1.0], [2.0]])  # init state

    x, u = use_modeling_tool(A, B, N, Q, R, P, x0, umax=umax, umin=umin)
    #  x, u = use_modeling_tool(A, B, N, Q, R, P, x0, umin=umin)

    rx1 = np.array(x[0, :]).flatten()
    rx2 = np.array(x[1, :]).flatten()
    ru = np.array(u[0, :]).flatten()

    flg, ax = plt.subplots(1)
    plt.plot(rx1, label="x1")
    plt.plot(rx2, label="x2")
    plt.plot(ru, label="u")
    plt.legend()
    plt.grid(True)

    x, u = hand_modeling(A, B, N, Q, R, P, x0, umax=umax, umin=umin)
    #  x, u = hand_modeling(A, B, N, Q, R, P, x0, umin=umin)
    x1 = np.array(x[:, 0]).flatten()
    x2 = np.array(x[:, 1]).flatten()

    #  flg, ax = plt.subplots(1)
    plt.plot(x1, '*r', label="x1")
    plt.plot(x2, '*b', label="x2")
    plt.plot(u, '*k', label="u")
    plt.legend()
    plt.grid(True)

    plt.show()


def test1():
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
    #  umax = 0.7

    x0 = np.matrix([[1.0], [2.0]])  # init state

    x, u = use_modeling_tool(A, B, N, Q, R, P, x0)

    rx1 = np.array(x[0, :]).flatten()
    rx2 = np.array(x[1, :]).flatten()
    ru = np.array(u[0, :]).flatten()

    flg, ax = plt.subplots(1)
    plt.plot(rx1, label="x1")
    plt.plot(rx2, label="x2")
    plt.plot(ru, label="u")
    plt.legend()
    plt.grid(True)

    x, u = hand_modeling(A, B, N, Q, R, P, x0)
    x1 = np.array(x[:, 0]).flatten()
    x2 = np.array(x[:, 1]).flatten()

    #  flg, ax = plt.subplots(1)
    plt.plot(x1, '*r', label="x1")
    plt.plot(x2, '*b', label="x2")
    plt.plot(u, '*k', label="u")
    plt.legend()
    plt.grid(True)

    plt.show()


def test3():
    print("start!!")
    A = np.matrix([[0.8, 1.0], [0, 0.9]])
    B = np.matrix([[-1.0], [2.0]])
    (nx, nu) = B.shape

    N = 10  # number of horizon
    Q = np.eye(nx)
    R = np.eye(nu)
    P = np.eye(nx)
    #  umax = 0.7
    #  umin = -0.7

    x0 = np.matrix([[1.0], [2.0]])  # init state

    xmin = np.matrix([[-3.5], [-0.5]])  # init state
    xmax = np.matrix([[3.5], [2.0]])  # init state

    #  x, u = use_modeling_tool(A, B, N, Q, R, P, x0, umax=umax, umin=umin, xmin=xmin, xmax=xmax)
    #  x, u = use_modeling_tool(A, B, N, Q, R, P, x0, umax=umax, umin=umin)
    x, u = use_modeling_tool(A, B, N, Q, R, P, x0)

    rx1 = np.array(x[0, :]).flatten()
    rx2 = np.array(x[1, :]).flatten()
    ru = np.array(u[0, :]).flatten()

    flg, ax = plt.subplots(1)
    plt.plot(rx1, label="x1")
    plt.plot(rx2, label="x2")
    plt.plot(ru, label="u")
    plt.legend()
    plt.grid(True)

    #  print(ru)

    x, u = hand_modeling2(A, B, N, Q, R, P, x0, xmin, xmax)
    #  x, u = hand_modeling(A, B, N, Q, R, P, x0, umax=umax, umin=umin)
    #  x, u = hand_modeling(A, B, N, Q, R, P, x0, umin=umin)
    x1 = np.array(x[0, :]).flatten()
    x2 = np.array(x[1, :]).flatten()
    u = np.array(u).flatten()

    #  flg, ax = plt.subplots(1)
    plt.plot(x1, '*r', label="x1")
    plt.plot(x2, '*b', label="x2")
    plt.plot(u, '*k', label="u")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    #  test1()
    #  test2()
    test3()
