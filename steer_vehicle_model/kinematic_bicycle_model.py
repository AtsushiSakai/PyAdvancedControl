#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Kinematic Bicycle Model

author Atsushi Sakai
"""

import math

dt = 0.1  # [s]
L = 5.0  # [m]
Lr = L / 2.0  # [m]


class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, beta=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.beta = beta


def update(state, a, delta):

    state.beta = math.atan2(Lr / L * math.tan(delta), 1.0)

    state.x = state.x + state.v * math.cos(state.yaw + state.beta) * dt
    state.y = state.y + state.v * math.sin(state.yaw + state.beta) * dt
    state.yaw = state.yaw + state.v / Lr * math.sin(state.beta) * dt
    state.v = state.v + a * dt

    return state


if __name__ == '__main__':
    print("start Kinematic Bicycle model simulation")
    import matplotlib.pyplot as plt
    import numpy as np

    T = 100
    a = [1.0] * T
    delta = [math.radians(1.0)] * T
    #  print(a, delta)

    state = State()

    x = []
    y = []
    yaw = []
    v = []
    beta = []
    time = []
    time = []
    t = 0.0

    for (ai, di) in zip(a, delta):
        t = t + dt
        state = update(state, ai, di)
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        beta.append(state.beta)
        time.append(t)

    flg, ax = plt.subplots(1)
    plt.plot(x, y)
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.axis("equal")
    plt.grid(True)

    flg, ax = plt.subplots(1)
    plt.plot(time, np.array(v) * 3.6)
    plt.xlabel("Time[km/h]")
    plt.ylabel("velocity[m]")
    plt.grid(True)

    #  flg, ax = plt.subplots(1)
    #  plt.plot([math.degrees(ibeta) for ibeta in beta])
    #  plt.grid(True)

    plt.show()
