#! /usr/bin/python
# -*- coding: utf-8 -*

import cvxpy
import numpy as np
import matplotlib.pyplot as plt


def get_nparray_from_matrix(x):
    u"""
    get build-in list from matrix
    """
    return np.array(x).flatten()


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

    x0 = np.matrix([[1], [2]])

    # mpc
    x = cvxpy.Variable(nx, N + 1)
    u = cvxpy.Variable(nu, N)

    costlist = 0.0
    constrlist = []

    for t in range(N):
        cost = 0.5 * cvxpy.quad_form(x[:, t], Q)
        cost += 0.5 * cvxpy.quad_form(u[:, t], R)
        #  cost += cvxpy.pos(u[t]) * R

        constr = [x[:, t + 1] == (A * x[:, t] + B * u[:, t])]

        costlist += cost
        constrlist += constr

    cost = 0.5 * cvxpy.quad_form(x[:, N], P)
    costlist += cost

    prob = cvxpy.Problem(cvxpy.Minimize(costlist), constrlist)

    prob.constraints += [x[:, 0] == x0]
    prob.constraints += [cvxpy.abs(u) <= umax]

    prob.solve(verbose=True)

    rx1 = get_nparray_from_matrix(x.value[0, :])
    rx2 = get_nparray_from_matrix(x.value[1, :])
    ru = get_nparray_from_matrix(u.value[0, :])

    flg, ax = plt.subplots(1)
    plt.plot(rx1, label="rx1")
    plt.plot(rx2, label="rx2")
    plt.plot(ru, label="ru")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
