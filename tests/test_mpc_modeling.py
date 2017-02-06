# -*- coding: utf-8 -*-
u"""
unittest code

author Atsushi Sakai
"""
import unittest
import numpy as np
import mpc_modeling.mpc_modeling


class Test(unittest.TestCase):

    def test_1(self):
        A = np.matrix([[0.8, 1.0], [0, 0.9]])
        B = np.matrix([[-1.0], [2.0]])
        (nx, nu) = B.shape

        N = 10  # number of horizon
        Q = np.eye(nx)
        R = np.eye(nu)
        P = np.eye(nx)

        x0 = np.matrix([[1.0], [2.0]])  # init state

        x, u = mpc_modeling.mpc_modeling.use_modeling_tool(A, B, N, Q, R, P, x0)

        rx1 = np.array(x[0, :]).flatten()
        rx2 = np.array(x[1, :]).flatten()
        ru = np.array(u[0, :]).flatten()

        x, u = mpc_modeling.mpc_modeling.hand_modeling(A, B, N, Q, R, P, x0)
        x1 = np.array(x[:, 0]).flatten()
        x2 = np.array(x[:, 1]).flatten()

        for (i, j) in zip(rx1, x1):
            print(i, j)
            assert (i - j) <= 0.0001, "Error"
        for (i, j) in zip(rx2, x2):
            print(i, j)
            assert (i - j) <= 0.0001, "Error"
        for (i, j) in zip(ru, u):
            print(i, j)
            assert (i - j) <= 0.0001, "Error"

    def test_2(self):
        A = np.matrix([[0.8, 1.0], [0, 0.9]])
        B = np.matrix([[-1.0], [2.0]])
        (nx, nu) = B.shape

        N = 10  # number of horizon
        Q = np.eye(nx)
        R = np.eye(nu)
        P = np.eye(nx)

        x0 = np.matrix([[1.0], [2.0]])  # init state
        umax = 0.7
        umin = 0.7

        x, u = mpc_modeling.mpc_modeling.use_modeling_tool(A, B, N, Q, R, P, x0, umax=umax, umin=umin)

        rx1 = np.array(x[0, :]).flatten()
        rx2 = np.array(x[1, :]).flatten()
        ru = np.array(u[0, :]).flatten()

        x, u = mpc_modeling.mpc_modeling.hand_modeling(A, B, N, Q, R, P, x0, umax=umax, umin=umin)

        x1 = np.array(x[:, 0]).flatten()
        x2 = np.array(x[:, 1]).flatten()

        for (i, j) in zip(rx1, x1):
            print(i, j)
            assert (i - j) <= 0.0001, "Error"
        for (i, j) in zip(rx2, x2):
            print(i, j)
            assert (i - j) <= 0.0001, "Error"
        for (i, j) in zip(ru, u):
            print(i, j)
            assert (i - j) <= 0.0001, "Error"

        x, u = mpc_modeling.mpc_modeling.use_modeling_tool(A, B, N, Q, R, P, x0, umax=umax)

        rx1 = np.array(x[0, :]).flatten()
        rx2 = np.array(x[1, :]).flatten()
        ru = np.array(u[0, :]).flatten()

        x, u = mpc_modeling.mpc_modeling.hand_modeling(A, B, N, Q, R, P, x0, umax=umax)

        x1 = np.array(x[:, 0]).flatten()
        x2 = np.array(x[:, 1]).flatten()

        for (i, j) in zip(rx1, x1):
            print(i, j)
            assert (i - j) <= 0.0001, "Error"
        for (i, j) in zip(rx2, x2):
            print(i, j)
            assert (i - j) <= 0.0001, "Error"
        for (i, j) in zip(ru, u):
            print(i, j)
            assert (i - j) <= 0.0001, "Error"

        x, u = mpc_modeling.mpc_modeling.use_modeling_tool(A, B, N, Q, R, P, x0, umin=umin)

        rx1 = np.array(x[0, :]).flatten()
        rx2 = np.array(x[1, :]).flatten()
        ru = np.array(u[0, :]).flatten()

        x, u = mpc_modeling.mpc_modeling.hand_modeling(A, B, N, Q, R, P, x0, umin=umin)

        x1 = np.array(x[:, 0]).flatten()
        x2 = np.array(x[:, 1]).flatten()

        for (i, j) in zip(rx1, x1):
            print(i, j)
            assert (i - j) <= 0.0001, "Error"
        for (i, j) in zip(rx2, x2):
            print(i, j)
            assert (i - j) <= 0.0001, "Error"
        for (i, j) in zip(ru, u):
            print(i, j)
            assert (i - j) <= 0.0001, "Error"

    def test_3(self):
        A = np.matrix([[0.8, 1.0], [0, 0.9]])
        B = np.matrix([[-1.0], [2.0]])
        (nx, nu) = B.shape

        N = 10  # number of horizon
        Q = np.eye(nx)
        R = np.eye(nu)
        P = np.eye(nx)

        x0 = np.matrix([[1.0], [2.0]])  # init state

        x, u = mpc_modeling.mpc_modeling.use_modeling_tool(A, B, N, Q, R, P, x0)

        rx1 = np.array(x[0, :]).flatten()
        rx2 = np.array(x[1, :]).flatten()
        ru = np.array(u[0, :]).flatten()

        x, u = mpc_modeling.mpc_modeling.hand_modeling2(A, B, N, Q, R, P, x0, None, None)

        x1 = np.array(x[0, :]).flatten()
        x2 = np.array(x[1, :]).flatten()
        u = np.array(u).flatten()

        for (i, j) in zip(rx1, x1):
            print(i, j)
            assert (i - j) <= 0.0001, "Error"
        for (i, j) in zip(rx2, x2):
            print(i, j)
            assert (i - j) <= 0.0001, "Error"
        for (i, j) in zip(ru, u):
            print(i, j)
            assert (i - j) <= 0.0001, "Error"


if __name__ == '__main__':
    unittest.main()
