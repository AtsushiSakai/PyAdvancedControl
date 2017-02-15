#! /usr/bin/python
# -*- coding: utf-8 -*
"""

author: Atsushi Sakai

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

from cvxopt import matrix
import cvxopt

DEBUG_ = False


def get_mat_psi(A, N):
    psi = np.matrix(np.zeros((0, A.shape[1])))

    for i in range(1, N + 1):
        psi = np.vstack((psi, A ** i))

    return psi


def get_mat_gamma(A, B, N):
    (nx, nu) = B.shape
    gamma = B

    for i in range(1, N):
        tmat = (A ** i) * B + gamma[-nx:, :]
        gamma = np.vstack((gamma, tmat))

    return gamma


def get_mat_theta(A, B, N):
    AiB = B
    (nx, nu) = B.shape
    theta = np.kron(np.eye(N), AiB)

    tmat = np.zeros((nx, 0))

    for i in range(1, N):
        t = np.zeros((nx, nu)) + B
        for ii in range(1, i + 1):
            t += (A ** ii) * B
        tmat = np.hstack((t, tmat))

    for i in range(1, N):
        theta[i * nx:(i + 1) * nx, :i] += tmat[:, -i:]

    return theta


def model_predictive_control(A, B, N, Q, R, T, x0, u0):

    (nx, nu) = B.shape

    du = np.matrix([0.0] * N).T

    psi = get_mat_psi(A, N)
    gamma = get_mat_gamma(A, B, N)
    theta = get_mat_theta(A, B, N)

    QQ = scipy.linalg.block_diag(np.kron(np.eye(N), Q))
    RR = scipy.linalg.block_diag(np.kron(np.eye(N), R))

    H = theta.T * QQ * theta + RR
    #  print(H)
    g = -2.0 * theta.T * QQ * (T - psi * x0 - gamma * u0)
    #  print(g)

    P = matrix(H)
    q = matrix(g)
    sol = cvxopt.solvers.qp(P, q)
    #  print(sol["x"])
    du = np.matrix(sol["x"])

    fx = psi * x0 + gamma * u0 + theta * du

    ffx = fx.reshape(N, nx)
    ffx = np.vstack((x0.T, ffx))
    #  print(ffx)

    u = np.cumsum(du).T

    return ffx, u


def test1():
    print("start!!")
    A = np.matrix([[0.8, 1.0], [0, 0.9]])
    B = np.matrix([[-1.0], [2.0]])
    (nx, nu) = B.shape

    N = 50  # number of horizon
    Q = np.diag([1, 1.0])
    R = np.eye(nu)

    x0 = np.matrix([2.0, 1.0]).T
    u0 = np.matrix([0.0])

    T = np.matrix([1.0, 0.25] * N).T
    #  print(T)

    x, u = model_predictive_control(A, B, N, Q, R, T, x0, u0)

    # test
    tx = x0
    rx = x0
    for iu in u[:, 0]:
        tx = A * tx + B * iu
        rx = np.hstack((rx, tx))

    if DEBUG_:
        plt.plot(x[:, 0])
        plt.plot(x[:, 1])
        plt.plot(u[:, 0])
        plt.grid(True)
        #  print(rx)
        plt.plot(rx[0, :].T, "xr")
        plt.plot(rx[1, :].T, "xb")

        plt.show()

    for ii in range(len(x[0, :]) + 1):
        for (i, j) in zip(rx[ii, :].T, x[:, ii]):
            assert (i - j) <= 0.0001, "Error" + str(i) + "," + str(j)


if __name__ == '__main__':
    DEBUG_ = True
    test1()
