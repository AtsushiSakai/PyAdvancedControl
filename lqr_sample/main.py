"""
Linear-Quadratic Regulator sample code

author Atsushi Sakai
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

simTime = 3.0
dt = 0.1

# x[k+1] = Ax[k] + Bu[k]
A = np.matrix([[1, 1.0], [0, 1]])
B = np.matrix([0.0, 1]).T
Q = np.matrix([[1.0, 0.0], [0.0, 0.0]])
R = np.matrix([[1.0]])
Kopt = None


def process(x, u):
    x = A * x + B * u
    return (x)


def solve_DARE_with_iteration(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE)
    """
    X = Q
    maxiter = 150
    eps = 0.01

    for i in range(maxiter):
        Xn = A.T * X * A - A.T * X * B * \
            la.inv(R + B.T * X * B) * B.T * X * A + Q
        if (abs(Xn - X)).max() < eps:
            X = Xn
            break
        X = Xn

    return Xn


def dlqr_with_iteration(Ad, Bd, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = Ad x[k] + Bd u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    X = solve_DARE_with_iteration(Ad, Bd, Q, R)

    # compute the LQR gain
    K = np.matrix(la.inv(Bd.T * X * Bd + R) * (Bd.T * X * Ad))

    return K


def dlqr_with_arimoto_potter(Ad, Bd, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = Ad x[k] + Bd u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    n = len(Bd)

    # continuous
    Ac = (Ad - np.eye(n)) / dt
    Bc = Bd / dt

    # Hamiltonian
    Ham = np.vstack(
        (np.hstack((Ac, - Bc * la.inv(R) * Bc.T)),
         np.hstack((-Q, -Ac.T))))

    eigVals, eigVecs = la.eig(Ham)

    V1 = None
    V2 = None

    for i in range(2 * n):
        if eigVals[i].real < 0:
            if V1 is None:
                V1 = eigVecs[0:n, i]
                V2 = eigVecs[n:2 * n, i]
            else:
                V1 = np.vstack((V1, eigVecs[0:n, i]))
                V2 = np.vstack((V2, eigVecs[n:2 * n, i]))
    V1 = np.matrix(V1.T)
    V2 = np.matrix(V2.T)

    P = (V2 * la.inv(V1)).real

    K = la.inv(R) * Bc.T * P

    return K


def lqr_regulator(x):
    global Kopt
    if Kopt is None:
        start = time.time()
        #  Kopt = dlqr_with_iteration(A, B, np.eye(2), np.eye(1))
        Kopt = dlqr_with_arimoto_potter(A, B, np.eye(2), np.eye(1))

        elapsed_time = time.time() - start
        print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    u = -Kopt * x
    return u


def lqr_ref_tracking(x, xref, uref):
    global Kopt
    if Kopt is None:
        #  start = time.time()
        #  Kopt = dlqr_with_iteration(A, B, np.eye(2), np.eye(1))
        Kopt = dlqr_with_arimoto_potter(A, B, Q, R)

        #  elapsed_time = time.time() - start
        #  print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    u = -uref - Kopt * (x - xref)

    return u


def main_regulator():
    t = 0.0

    x = np.matrix([3, 1]).T
    u = np.matrix([0])

    time_history = [0.0]
    x1_history = [x[0, 0]]
    x2_history = [x[1, 0]]
    u_history = [0.0]

    while t <= simTime:
        u = lqr_regulator(x)

        u0 = float(u[0, 0])
        x = process(x, u0)

        x1_history.append(x[0, 0])
        x2_history.append(x[1, 0])

        u_history.append(u0)
        time_history.append(t)
        t += dt

    plt.plot(time_history, u_history, "-r", label="input")
    plt.plot(time_history, x1_history, "-b", label="x1")
    plt.plot(time_history, x2_history, "-g", label="x2")
    plt.grid(True)
    plt.xlim([0, simTime])
    plt.title("LQR Regulator")
    plt.legend()
    plt.show()


def main_reference_tracking():
    t = 0.0

    x = np.matrix([3, 1]).T
    u = np.matrix([0])
    xref = np.matrix([1, 0]).T
    uref = 0.0

    time_history = [0.0]
    x1_history = [x[0, 0]]
    x2_history = [x[1, 0]]
    u_history = [0.0]

    while t <= simTime:
        u = lqr_ref_tracking(x, xref, uref)

        u0 = float(u[0, 0])
        x = process(x, u0)

        x1_history.append(x[0, 0])
        x2_history.append(x[1, 0])

        u_history.append(u0)
        time_history.append(t)
        t += dt

    plt.plot(time_history, u_history, "-r", label="input")
    plt.plot(time_history, x1_history, "-b", label="x1")
    plt.plot(time_history, x2_history, "-g", label="x2")
    xref0_h = [xref[0, 0] for i in range(len(time_history))]
    xref1_h = [xref[1, 0] for i in range(len(time_history))]
    plt.plot(time_history, xref0_h, "--b", label="target x1")
    plt.plot(time_history, xref1_h, "--g", label="target x2")

    plt.grid(True)
    plt.xlim([0, simTime])
    plt.title("LQR Tracking")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    print("Start")
    #  main_regulator()
    main_reference_tracking()
    print("Done")
