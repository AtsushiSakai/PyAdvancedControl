#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Interface function to solve a quadratic programing problem with ECOS.

Author: Atsushi Sakai

"""


import numpy as np
import math
import ecos
import scipy.linalg
import scipy.sparse as sp


def ecosqp(H, f, A=None, B=None, Aeq=None, Beq=None):
    """
    solve a quadratic programing problem with ECOS

        min 1/2*x'*H*x + f'*x
        s.t. A*x <= b
             Aeq*x = beq

    return sol
        It is same data format of CVXOPT.

    """
    # ===dimension and argument checking===
    # H
    assert H.shape[0] == H.shape[1], "Hessian must be a square matrix"

    n = H.shape[0]

    # f
    if (f is None) or (f.size == 0):
        f = np.zeros((n, 1))
    else:
        assert f.shape[0] == n, "Linear term f must be a column vector of length"
        assert f.shape[1] == 1, "Linear term f must be a column vector"

    # check cholesky
    try:
        W = np.linalg.cholesky(H)
    except np.linalg.linalg.LinAlgError:
        W = scipy.linalg.sqrtm(H)
    #  print(W)

    # set up SOCP problem
    c = np.vstack((np.zeros((n, 1)), 1.0))
    #  print(c)

    # pad Aeq with a zero column for t
    if Aeq is not None:
        Aeq = np.hstack((Aeq, np.zeros((Aeq.shape[0], 1))))
        beq = Beq
    else:
        Aeq = np.matrix([])
        beq = np.matrix([])

    # create second-order cone constraint for objective function
    fhalf = f / math.sqrt(2.0)
    #  print(fhalf)
    zerocolumn = np.zeros((W.shape[1], 1))
    #  print(zerocolumn)

    tmp = 1.0 / math.sqrt(2.0)

    Gquad1 = np.hstack((fhalf.T, np.matrix(-tmp)))
    Gquad2 = np.hstack((-W, zerocolumn))
    Gquad3 = np.hstack((-fhalf.T, np.matrix(tmp)))
    Gquad = np.vstack((Gquad1, Gquad2, Gquad3))
    #  print(Gquad1)
    #  print(Gquad2)
    #  print(Gquad3)
    #  print(Gquad)

    hquad = np.vstack((tmp, zerocolumn, tmp))
    #  print(hquad)

    if A is None:
        G = Gquad
        h = hquad
        dims = {'q': [W.shape[1] + 2], 'l': 0}
    else:
        G1 = np.hstack((A, np.zeros((A.shape[0], 1))))
        G = np.vstack((G1, Gquad))
        h = np.vstack((B, hquad))
        dims = {'q': [W.shape[1] + 2], 'l': A.shape[0]}

    c = np.array(c).flatten()
    G = sp.csc_matrix(G)
    h = np.array(h).flatten()

    if Aeq.size == 0:
        sol = ecos.solve(c, G, h, dims)
    else:
        Aeq = sp.csc_matrix(Aeq)
        beq = np.array(beq).flatten()
        sol = ecos.solve(c, G, h, dims, Aeq, beq)
    #  print(sol)
    #  print(sol["x"])

    sol["fullx"] = sol["x"]
    sol["x"] = sol["fullx"][:n]
    sol["fval"] = sol["fullx"][-1]

    return sol


def test1():
    import cvxopt
    from cvxopt import matrix

    P = matrix(np.diag([1.0, 0.0]))
    q = matrix(np.array([3.0, 4.0]).T)
    G = matrix(np.array([[-1.0, 0.0], [0, -1.0], [-1.0, -3.0], [2.0, 5.0], [3.0, 4.0]]))
    h = matrix(np.array([0.0, 0.0, -15.0, 100.0, 80.0]).T)

    sol = cvxopt.solvers.qp(P, q, G, h)

    #  print(sol)
    print(sol["x"])
    #  print(sol["primal objective"])

    assert sol["x"][0] - 0.0, "Error1"
    assert sol["x"][1] - 5.0, "Error2"

    P = np.diag([1.0, 0.0])
    q = np.matrix([3.0, 4.0]).T
    G = np.matrix([[-1.0, 0.0], [0, -1.0], [-1.0, -3.0], [2.0, 5.0], [3.0, 4.0]])
    h = np.matrix([0.0, 0.0, -15.0, 100.0, 80.0]).T

    sol2 = ecosqp(P, q, G, h)

    for i in range(len(sol["x"])):
        assert (sol["x"][i] - sol2["x"][i]) <= 0.001, "Error1"


def test2():
    import cvxopt
    from cvxopt import matrix

    P = np.matrix([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 1., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 1., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 1., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 1., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 1., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 1.]])
    #  print(P.shape)

    q = np.matrix([[0.],
                   [0.],
                   [0.],
                   [0.],
                   [0.],
                   [0.],
                   [0.],
                   [0.],
                   [0.]])
    #  print(q.shape)

    A = np.matrix([[1., 0., 0., 1., 0., 0., 0., 0., 0.],
                   [-2., -0., -0., 0., 1., 0., 0., 0., 0.],
                   [0., 1., 0., -0.8, -1., 1., 0., 0., 0.],
                   [-0., -2., -0., 0., -0.9, 0., 1., 0., 0.],
                   [0., 0., 1., 0., 0., -0.8, -1., 1., 0.],
                   [-0., -0., -2., 0., 0., 0., -0.9, 0., 1.]])
    #  print(A.shape)

    B = np.matrix([[2.8],
                   [1.8],
                   [0.],
                   [0.],
                   [0.],
                   [0.]])
    #  print(B.shape)

    sol = cvxopt.solvers.qp(matrix(P), matrix(q), A=matrix(A), b=matrix(B))

    #  #  print(sol)
    print(sol["x"])
    #  #  print(sol["primal objective"])

    sol2 = ecosqp(P, q, Aeq=A, Beq=B)
    print(sol2["x"])

    for i in range(len(sol["x"])):
        print(sol["x"][i], sol2["x"][i])
        assert (sol["x"][i] - sol2["x"][i]) <= 0.001, "Error1"


def test3():
    import cvxopt
    from cvxopt import matrix

    P = np.matrix([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 1., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 1., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 1., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 1., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 1., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 1.]])
    #  print(P.shape)

    q = np.matrix([[0.],
                   [0.],
                   [0.],
                   [0.],
                   [0.],
                   [0.],
                   [0.],
                   [0.],
                   [0.]])
    #  print(q.shape)

    A = np.matrix([[1., 0., 0., 1., 0., 0., 0., 0., 0.],
                   [-2., -0., -0., 0., 1., 0., 0., 0., 0.],
                   [0., 1., 0., -0.8, -1., 1., 0., 0., 0.],
                   [-0., -2., -0., 0., -0.9, 0., 1., 0., 0.],
                   [0., 0., 1., 0., 0., -0.8, -1., 1., 0.],
                   [-0., -0., -2., 0., 0., 0., -0.9, 0., 1.]])
    #  print(A.shape)

    B = np.matrix([[2.8],
                   [1.8],
                   [0.],
                   [0.],
                   [0.],
                   [0.]])
    #  print(B.shape)

    G = np.matrix([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 1., 0., 0., 0., 0., 0., 0.],
                   [-1., -0., -0., 0., 0., 0., 0., 0., 0.],
                   [-0., -1., -0., 0., 0., 0., 0., 0., 0.],
                   [-0., -0., -1., 0., 0., 0., 0., 0., 0.]])
    print(G)

    h = np.matrix([[0.7],
                   [0.7],
                   [0.7],
                   [0.7],
                   [0.7],
                   [0.7]])

    print(h)

    sol = cvxopt.solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), A=matrix(A), b=matrix(B))

    #  #  print(sol)
    print(sol["x"])
    #  #  print(sol["primal objective"])

    sol2 = ecosqp(P, q, A=G, B=h, Aeq=A, Beq=B)
    print(sol2["x"])

    for i in range(len(sol["x"])):
        print(sol["x"][i], sol2["x"][i])
        assert (sol["x"][i] - sol2["x"][i]) <= 0.001, "Error1"


if __name__ == '__main__':
    #  test1()
    #  test2()
    test3()
