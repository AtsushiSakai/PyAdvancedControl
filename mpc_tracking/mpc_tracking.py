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


def get_mat_psi(A, N):
    psi = np.matrix(np.zeros((0, A.shape[1])))

    for i in range(1, N + 1):
        psi = np.vstack((psi, A ** i))

    return psi


def get_mat_gamma(A, B, N):

    gamma = B

    for i in range(1, N):
        gamma = np.vstack((gamma, (A ** i) * B + gamma[-1, :]))

    #  print(gamma)

    return gamma


def get_mat_theta(A, B, N):
    AiB = B
    theta = np.kron(np.eye(N), AiB)
    for i in range(1, N):
        AiB = A * AiB
        theta += np.kron(np.diag(np.ones(N - i), -i), AiB)

    #  print(theta)
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

    plt.plot(ffx[:, 0])
    plt.plot(ffx[:, 1])
    plt.grid(True)
    plt.show()


def test1():
    print("start!!")
    A = np.matrix([[0.8, 1.0], [0, 0.9]])
    B = np.matrix([[-1.0], [2.0]])
    (nx, nu) = B.shape

    N = 100  # number of horizon
    Q = np.diag([1, 1.0])
    R = np.eye(nu)

    x0 = np.matrix([2.0, 1.0]).T
    u0 = np.matrix([0.0])

    T = np.matrix([1.0, 0.25] * N).T
    #  print(T)

    model_predictive_control(A, B, N, Q, R, T, x0, u0)


if __name__ == '__main__':
    DEBUG_ = True
    test1()
