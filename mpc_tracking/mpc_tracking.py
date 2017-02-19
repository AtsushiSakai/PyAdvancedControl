#! /usr/bin/python
# -*- coding: utf-8 -*
"""

MPC tracking sample code

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
    gamma = np.zeros((nx, nu)) + B

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


def generate_du_constraints_mat(G, h, N, nu, mindu, maxdu):

    if maxdu is not None:
        tG = np.matrix(np.eye(N * nu))
        th = np.kron(np.ones((N * nu, 1)), maxdu)
        G = np.vstack([G, tG])
        h = np.vstack([h, th])

    if mindu is not None:
        tG = np.matrix(np.eye(N * nu)) * -1.0
        th = np.kron(np.ones((N * nu, 1)), mindu * -1.0)
        G = np.vstack([G, tG])
        h = np.vstack([h, th])

    return G, h


def generate_u_constraints_mat(G, h, N, nu, u0, minu, maxu):

    # calc F
    F = np.matrix(np.zeros((0, N * nu + 1)))
    for i in range(N * nu):
        if maxu is not None:
            tF = np.zeros((1, N * nu + 1))
            tF[0, i] = 1.0 / maxu
            tF[0, -1] = -1.0
            F = np.vstack([F, tF])

        if minu is not None:
            tF = np.zeros((1, N * nu + 1))
            tF[0, i] = 1.0 / minu
            tF[0, -1] = -1.0
            F = np.vstack([F, tF])

    f = F[:, -1]
    #  print(f)

    # calc SF
    SF = np.matrix(np.zeros((F.shape[0], F.shape[1] - 1)))
    for i in range(0, N * nu):
        for ii in range(i, N * nu):
            SF[:, i] += F[:, ii]
    #  print(SF)

    SF1 = SF[:, 0]
    #  print(SF1)

    th = -SF1 * u0 - f
    #  print(th)

    G = np.vstack([G, SF])
    h = np.vstack([h, th])

    return G, h


def generate_x_constraints_mat(G, h, N, nx, u0, theta, kappa, minx, maxx):

    # calc G
    cG = np.matrix(np.zeros((0, N * nx + 1)))
    for i in range(N):
        if maxx is not None:
            for ii in range(maxx.shape[1]):
                tG = np.zeros((1, N * nx + 1))
                tG[0, i * nx + ii] = 1.0 / maxx[0, ii]
                tG[0, -1] = -1.0
                cG = np.vstack([cG, tG])

        if minx is not None:
            for ii in range(minx.shape[1]):
                tG = np.zeros((1, N * nx + 1))
                tG[0, i * nx + ii] = 1.0 / minx[0, ii]
                tG[0, -1] = -1.0
                cG = np.vstack([cG, tG])

    print(cG)

    tau = cG[:, :-1]
    g = cG[:, -1]
    #  print(tau)
    #  print(g)

    tmpG = tau * theta
    tmph = -tau * kappa - g
    #  print(tmpG)
    #  print(tmph)

    #  print(cG)

    G = np.vstack([G, tmpG])
    h = np.vstack([h, tmph])

    return G, h


def model_predictive_control(A, B, N, Q, R, T, x0, u0, mindu=None, maxdu=None, minu=None, maxu=None, maxx=None, minx=None):

    (nx, nu) = B.shape

    du = np.matrix([0.0] * N).T

    psi = get_mat_psi(A, N)
    gamma = get_mat_gamma(A, B, N)
    theta = get_mat_theta(A, B, N)

    QQ = scipy.linalg.block_diag(np.kron(np.eye(N), Q))
    RR = scipy.linalg.block_diag(np.kron(np.eye(N), R))

    H = theta.T * QQ * theta + RR
    g = - theta.T * QQ * (T - psi * x0 - gamma * u0)
    #  print(H)
    #  print(g)
    #  print(u0)

    G = np.zeros((0, nu * N))
    h = np.zeros((0, nu))

    G, h = generate_du_constraints_mat(G, h, N, nu, mindu, maxdu)
    G, h = generate_u_constraints_mat(G, h, N, nu, u0, minu, maxu)

    kappa = psi * x0 + gamma * u0
    G, h = generate_x_constraints_mat(G, h, N, nx, u0, theta, kappa, minx, maxx)

    P = matrix(H)
    q = matrix(g)
    G = matrix(G)
    h = matrix(h)
    sol = cvxopt.solvers.qp(P, q, G, h)

    du = np.matrix(sol["x"])
    #  print(du)
    #  print(len(du))

    fx = psi * x0 + gamma * u0 + theta * du

    ffx = fx.reshape(N, nx)
    ffx = np.vstack((x0.T, ffx))
    #  print(ffx)

    u = np.cumsum(du).T + u0
    #  print(u)

    return ffx, u, du


def test1():
    print("start!!")
    A = np.matrix([[0.8, 1.0], [0, 0.9]])
    B = np.matrix([[-1.0], [2.0]])
    (nx, nu) = B.shape

    N = 50  # number of horizon
    Q = np.diag([1.0, 1.0])
    R = np.eye(nu)

    x0 = np.matrix([2.0, 1.0]).T
    u0 = np.matrix([0.0])

    T = np.matrix([1.0, 0.25] * N).T
    #  print(T)

    x, u, du = model_predictive_control(A, B, N, Q, R, T, x0, u0)

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

    target = T.reshape(N, nx)
    for ii in range(len(x[0, :]) + 1):
        assert abs(x[-1, ii] - target[-1, ii]) <= 0.3, "Error"


def test2():
    print("start!!")
    A = np.matrix([[0.8, 1.0], [0, 0.9]])
    B = np.matrix([[-1.0], [2.0]])
    (nx, nu) = B.shape

    N = 50  # number of horizon
    Q = np.diag([1.0, 1.0])
    R = np.eye(nu)

    x0 = np.matrix([2.0, 1.0]).T
    u0 = np.matrix([0.1])

    T = np.matrix([1.0, 0.25] * N).T
    #  print(T)

    x, u, du = model_predictive_control(A, B, N, Q, R, T, x0, u0)

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

    target = T.reshape(N, nx)
    for ii in range(len(x[0, :]) + 1):
        assert abs(x[-1, ii] - target[-1, ii]) <= 0.3, "Error"


def test3():
    print("start!!")
    A = np.matrix([[0.8, 1.0], [0, 0.9]])
    B = np.matrix([[-1.0], [2.0]])
    (nx, nu) = B.shape

    N = 50  # number of horizon
    Q = np.diag([1.0, 1.0])
    R = np.eye(nu)

    x0 = np.matrix([2.0, 1.0]).T
    u0 = np.matrix([-0.1])

    T = np.matrix([1.0, 0.25] * N).T
    #  print(T)

    x, u, du = model_predictive_control(A, B, N, Q, R, T, x0, u0)

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

    target = T.reshape(N, nx)
    for ii in range(len(x[0, :]) + 1):
        assert abs(x[-1, ii] - target[-1, ii]) <= 0.3, "Error"


def test4():
    print("start!!")
    A = np.matrix([[0.8, 1.0], [0, 0.9]])
    B = np.matrix([[-1.0], [2.0]])
    (nx, nu) = B.shape

    N = 50  # number of horizon
    Q = np.diag([1.0, 1.0])
    R = np.eye(nu)

    x0 = np.matrix([0.0, 1.0]).T
    u0 = np.matrix([-0.1])

    T = np.matrix([1.0, 0.25] * N).T
    #  print(T)

    x, u, du = model_predictive_control(A, B, N, Q, R, T, x0, u0)

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

    target = T.reshape(N, nx)
    for ii in range(len(x[0, :]) + 1):
        assert abs(x[-1, ii] - target[-1, ii]) <= 0.3, "Error"


def test5():
    print("start!!")
    A = np.matrix([[0.8, 1.0], [0, 0.9]])
    B = np.matrix([[-1.0], [2.0]])
    (nx, nu) = B.shape

    N = 30  # number of horizon
    Q = np.diag([1.0, 1.0])
    R = np.eye(nu)

    x0 = np.matrix([0.0, -1.0]).T
    u0 = np.matrix([-0.1])

    T = np.matrix([1.0, 0.25] * N).T
    #  print(T)

    x, u, du = model_predictive_control(A, B, N, Q, R, T, x0, u0)

    # test
    tx = x0
    rx = x0
    for iu in u[:, 0]:
        tx = A * tx + B * iu
        rx = np.hstack((rx, tx))

    if DEBUG_:
        plt.plot(x[:, 0], label="x1")
        plt.plot(x[:, 1], label="x2")
        plt.plot(u[:, 0], label="u")
        plt.grid(True)
        #  print(rx)
        plt.plot(rx[0, :].T, "xr", label="model x1")
        plt.plot(rx[1, :].T, "xb", label="model x2")

        plt.legend()

        plt.show()

    for ii in range(len(x[0, :]) + 1):
        for (i, j) in zip(rx[ii, :].T, x[:, ii]):
            assert (i - j) <= 0.0001, "Error" + str(i) + "," + str(j)

    target = T.reshape(N, nx)
    for ii in range(len(x[0, :]) + 1):
        assert abs(x[-1, ii] - target[-1, ii]) <= 0.3, "Error"


def test6():
    print("start!!")
    A = np.matrix([[0.8, 1.0], [0, 0.9]])
    B = np.matrix([[-1.0], [2.0]])
    (nx, nu) = B.shape

    N = 30  # number of horizon
    Q = np.diag([1.0, 1.0])
    R = np.eye(nu)

    x0 = np.matrix([0.0, -1.0]).T
    u0 = np.matrix([-0.1])

    T = np.matrix([1.0, 0.25] * N).T
    #  print(T)

    mindu = -0.5
    maxdu = 0.5

    x, u, du = model_predictive_control(A, B, N, Q, R, T, x0, u0, mindu=mindu, maxdu=maxdu)

    # test
    tx = x0
    rx = x0
    for iu in u[:, 0]:
        tx = A * tx + B * iu
        rx = np.hstack((rx, tx))

    if DEBUG_:
        plt.plot(x[:, 0], label="x1")
        plt.plot(x[:, 1], label="x2")
        plt.plot(u[:, 0], label="u")
        plt.plot(du, label="du")
        plt.grid(True)
        #  print(rx)
        plt.plot(rx[0, :].T, "xr", label="model x1")
        plt.plot(rx[1, :].T, "xb", label="model x2")

        plt.legend()

        plt.show()

    for ii in range(len(x[0, :]) + 1):
        for (i, j) in zip(rx[ii, :].T, x[:, ii]):
            assert (i - j) <= 0.0001, "Error" + str(i) + "," + str(j)

    target = T.reshape(N, nx)
    for ii in range(len(x[0, :]) + 1):
        assert abs(x[-1, ii] - target[-1, ii]) <= 0.3, "Error"

    for i in du:
        assert i <= maxdu + 0.0001, "Error" + str(i) + "," + str(maxdu)
        assert i >= mindu - 0.0001, "Error" + str(i) + "," + str(mindu)


def test7():
    print("start!!")
    A = np.matrix([[0.8, 1.0], [0, 0.9]])
    B = np.matrix([[-1.0], [2.0]])
    (nx, nu) = B.shape

    N = 30  # number of horizon
    Q = np.diag([1.0, 1.0])
    R = np.eye(nu)

    x0 = np.matrix([0.0, -1.0]).T
    u0 = np.matrix([-0.1])

    T = np.matrix([1.0, 0.25] * N).T
    #  print(T)

    maxdu = 0.2
    mindu = -0.3

    x, u, du = model_predictive_control(A, B, N, Q, R, T, x0, u0, mindu=mindu, maxdu=maxdu)

    # test
    tx = x0
    rx = x0
    for iu in u[:, 0]:
        tx = A * tx + B * iu
        rx = np.hstack((rx, tx))

    if DEBUG_:
        plt.plot(x[:, 0], label="x1")
        plt.plot(x[:, 1], label="x2")
        plt.plot(u[:, 0], label="u")
        plt.plot(du, label="du")
        plt.grid(True)
        #  print(rx)
        plt.plot(rx[0, :].T, "xr", label="model x1")
        plt.plot(rx[1, :].T, "xb", label="model x2")

        plt.legend()

        plt.show()

    for ii in range(len(x[0, :]) + 1):
        for (i, j) in zip(rx[ii, :].T, x[:, ii]):
            assert (i - j) <= 0.0001, "Error" + str(i) + "," + str(j)

    target = T.reshape(N, nx)
    for ii in range(len(x[0, :]) + 1):
        assert abs(x[-1, ii] - target[-1, ii]) <= 0.3, "Error"

    for i in du:
        assert i <= maxdu + 0.0001, "Error" + str(i) + "," + str(maxdu)
        assert i >= mindu - 0.0001, "Error" + str(i) + "," + str(mindu)


def test8():
    print("start!!")
    A = np.matrix([[0.8, 1.0], [0, 0.9]])
    B = np.matrix([[-1.0], [2.0]])
    (nx, nu) = B.shape

    N = 30  # number of horizon
    Q = np.diag([1.0, 1.0])
    R = np.eye(nu)

    x0 = np.matrix([0.0, -1.0]).T
    u0 = np.matrix([-0.1])

    T = np.matrix([0.0, 0.0] * N).T
    #  print(T)

    maxdu = 0.2
    mindu = -0.3

    x, u, du = model_predictive_control(A, B, N, Q, R, T, x0, u0, mindu=mindu, maxdu=maxdu)

    # test
    tx = x0
    rx = x0
    for iu in u[:, 0]:
        tx = A * tx + B * iu
        rx = np.hstack((rx, tx))

    if DEBUG_:
        plt.plot(x[:, 0], label="x1")
        plt.plot(x[:, 1], label="x2")
        plt.plot(u[:, 0], label="u")
        plt.plot(du, label="du")
        plt.grid(True)
        #  print(rx)
        plt.plot(rx[0, :].T, "xr", label="model x1")
        plt.plot(rx[1, :].T, "xb", label="model x2")

        plt.legend()

        plt.show()

    for ii in range(len(x[0, :]) + 1):
        for (i, j) in zip(rx[ii, :].T, x[:, ii]):
            assert (i - j) <= 0.0001, "Error" + str(i) + "," + str(j)

    target = T.reshape(N, nx)
    for ii in range(len(x[0, :]) + 1):
        assert abs(x[-1, ii] - target[-1, ii]) <= 0.3, "Error"

    for i in du:
        assert i <= maxdu + 0.0001, "Error" + str(i) + "," + str(maxdu)
        assert i >= mindu - 0.0001, "Error" + str(i) + "," + str(mindu)


def test9():
    print("start!!")
    A = np.matrix([[0.8, 1.0], [0, 0.9]])
    B = np.matrix([[-1.0], [2.0]])
    (nx, nu) = B.shape

    N = 30  # number of horizon
    Q = np.diag([1.0, 1.0])
    R = np.eye(nu)

    x0 = np.matrix([0.0, -1.0]).T
    u0 = np.matrix([-0.1])

    T = np.matrix([0.0, 0.0] * N).T
    #  print(T)

    maxu = 0.1
    minu = -0.3

    x, u, du = model_predictive_control(A, B, N, Q, R, T, x0, u0, minu=minu, maxu=maxu)

    # test
    tx = x0
    rx = x0
    for iu in u[:, 0]:
        tx = A * tx + B * iu
        rx = np.hstack((rx, tx))

    if DEBUG_:
        plt.plot(x[:, 0], label="x1")
        plt.plot(x[:, 1], label="x2")
        plt.plot(u[:, 0], label="u")
        plt.plot(du, label="du")
        plt.grid(True)
        #  print(rx)
        plt.plot(rx[0, :].T, "xr", label="model x1")
        plt.plot(rx[1, :].T, "xb", label="model x2")

        plt.legend()

        plt.show()

    for ii in range(len(x[0, :]) + 1):
        for (i, j) in zip(rx[ii, :].T, x[:, ii]):
            assert (i - j) <= 0.0001, "Error" + str(i) + "," + str(j)

    target = T.reshape(N, nx)
    for ii in range(len(x[0, :]) + 1):
        assert abs(x[-1, ii] - target[-1, ii]) <= 0.3, "Error"

    for i in u:
        assert i <= maxu + 0.0001, "Error" + str(i) + "," + str(maxu)
        assert i >= minu - 0.0001, "Error" + str(i) + "," + str(minu)


def test10():
    print("start!!")
    A = np.matrix([[0.8, 1.0], [0, 0.9]])
    B = np.matrix([[-1.0], [2.0]])
    (nx, nu) = B.shape

    N = 30  # number of horizon
    Q = np.diag([1.0, 1.0])
    R = np.eye(nu)

    x0 = np.matrix([0.0, -1.0]).T
    u0 = np.matrix([-0.1])

    T = np.matrix([0.0, 0.0] * N).T
    #  print(T)

    #  maxu = 0.1
    minu = -0.0001

    x, u, du = model_predictive_control(A, B, N, Q, R, T, x0, u0, minu=minu)

    # test
    tx = x0
    rx = x0
    for iu in u[:, 0]:
        tx = A * tx + B * iu
        rx = np.hstack((rx, tx))

    if DEBUG_:
        plt.plot(x[:, 0], label="x1")
        plt.plot(x[:, 1], label="x2")
        plt.plot(u[:, 0], label="u")
        plt.plot(du, label="du")
        plt.grid(True)
        #  print(rx)
        plt.plot(rx[0, :].T, "xr", label="model x1")
        plt.plot(rx[1, :].T, "xb", label="model x2")

        plt.legend()

        plt.show()

    for ii in range(len(x[0, :]) + 1):
        for (i, j) in zip(rx[ii, :].T, x[:, ii]):
            assert (i - j) <= 0.0001, "Error" + str(i) + "," + str(j)

    target = T.reshape(N, nx)
    for ii in range(len(x[0, :]) + 1):
        assert abs(x[-1, ii] - target[-1, ii]) <= 0.3, "Error"

    for i in u:
        #  assert i <= maxu + 0.0001, "Error" + str(i) + "," + str(maxu)
        assert i >= minu - 0.0001, "Error" + str(i) + "," + str(minu)


def test11():
    print("start!!")
    A = np.matrix([[0.8, 1.0], [0, 0.9]])
    B = np.matrix([[-1.0], [2.0]])
    (nx, nu) = B.shape

    N = 30  # number of horizon
    Q = np.diag([1.0, 1.0])
    R = np.eye(nu)

    x0 = np.matrix([0.0, -1.0]).T
    u0 = np.matrix([-0.1])

    T = np.matrix([0.0, 0.0] * N).T
    #  print(T)

    maxx = np.matrix([0.5, 0.2])
    minx = np.matrix([-1.5, -2.5])

    x, u, du = model_predictive_control(A, B, N, Q, R, T, x0, u0, maxx=maxx, minx=minx)

    # test
    tx = x0
    rx = x0
    for iu in u[:, 0]:
        tx = A * tx + B * iu
        rx = np.hstack((rx, tx))

    if DEBUG_:
        plt.plot(x[:, 0], label="x1")
        plt.plot(x[:, 1], label="x2")
        plt.plot(u[:, 0], label="u")
        plt.plot(du, label="du")
        plt.grid(True)
        #  print(rx)
        plt.plot(rx[0, :].T, "xr", label="model x1")
        plt.plot(rx[1, :].T, "xb", label="model x2")

        plt.legend()

        plt.show()

    for ii in range(len(x[0, :]) + 1):
        for (i, j) in zip(rx[ii, :].T, x[:, ii]):
            assert (i - j) <= 0.0001, "Error" + str(i) + "," + str(j)

    target = T.reshape(N, nx)
    for ii in range(len(x[0, :]) + 1):
        assert abs(x[-1, ii] - target[-1, ii]) <= 0.3, "Error"

    for ii in range(len(x[:, 0])):
        for i in range(len(x[0, :])):
            assert x[ii, i] <= maxx[i, 0], "Error" + str(x[ii, i]) + "," + str(maxx[i, 0])
            assert x[ii, i] >= minx[i, 0], "Error" + str(x[ii, i]) + "," + str(minx[i, 0])


def test12():
    print("start!!")
    A = np.matrix([[0.8, 1.0], [0, 0.9]])
    B = np.matrix([[-1.0], [2.0]])
    (nx, nu) = B.shape

    N = 30  # number of horizon
    Q = np.diag([1.0, 1.0])
    R = np.eye(nu)

    x0 = np.matrix([0.0, -1.0]).T
    u0 = np.matrix([-0.1])

    T = np.matrix([0.0, 0.0] * N).T
    #  print(T)

    maxx = np.matrix([0.5, 0.2])
    minx = np.matrix([-1.5, -2.5])

    mindu = -0.5
    maxdu = 0.5

    x, u, du = model_predictive_control(A, B, N, Q, R, T, x0, u0, maxx=maxx, minx=minx, maxdu=maxdu, mindu=mindu)

    # test
    tx = x0
    rx = x0
    for iu in u[:, 0]:
        tx = A * tx + B * iu
        rx = np.hstack((rx, tx))

    if DEBUG_:
        plt.plot(x[:, 0], label="x1")
        plt.plot(x[:, 1], label="x2")
        plt.plot(u[:, 0], label="u")
        plt.plot(du, label="du")
        plt.grid(True)
        #  print(rx)
        plt.plot(rx[0, :].T, "xr", label="model x1")
        plt.plot(rx[1, :].T, "xb", label="model x2")

        plt.legend()

        plt.show()

    for ii in range(len(x[0, :]) + 1):
        for (i, j) in zip(rx[ii, :].T, x[:, ii]):
            assert (i - j) <= 0.0001, "Error" + str(i) + "," + str(j)

    target = T.reshape(N, nx)
    for ii in range(len(x[0, :]) + 1):
        assert abs(x[-1, ii] - target[-1, ii]) <= 0.3, "Error"

    for ii in range(len(x[:, 0])):
        for i in range(len(x[0, :])):
            assert x[ii, i] <= maxx[i, 0], "Error" + str(x[ii, i]) + "," + str(maxx[i, 0])
            assert x[ii, i] >= minx[i, 0], "Error" + str(x[ii, i]) + "," + str(minx[i, 0])

    for i in du:
        assert i <= maxdu + 0.0001, "Error" + str(i) + "," + str(maxdu)
        assert i >= mindu - 0.0001, "Error" + str(i) + "," + str(mindu)


if __name__ == '__main__':
    DEBUG_ = True
    #  test1()
    #  test2()
    #  test3()
    #  test4()
    #  test5()
    #  test6()
    #  test7()
    #  test8()
    #  test9()
    #  test10()
    #  test11()
    test12()
