#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Dynamic Bicycle Model

author Atsushi Sakai
"""

import math

dt = 0.1  # [s]
L = 5.0  # [m]
Lr = L / 2.0  # [m]
Lf = L - Lr
Cf = 1600.0 * 2.0  # N/rad
Cr = 1700.0 * 2.0  # N/rad
Iz = 2250.0  # kg/m2
m = 1500.0  # kg


class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, vx=0.01, vy=0.0, omega=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vx = vx
        self.vy = vy
        self.omega = omega


def update(s, a, delta):
    s.x = s.x + s.vx * math.cos(s.yaw) * dt - s.vy * math.sin(s.yaw) * dt
    s.y = s.y + s.vx * math.sin(s.yaw) * dt + s.vy * math.cos(s.yaw) * dt
    s.yaw = s.yaw + s.omega * dt
    Ffy = -Cf * math.atan2(((s.vy + Lf * s.omega) / s.vx - delta), 1.0)
    Fry = -Cr * math.atan2((s.vy - Lr * s.omega) / s.vx, 1.0)
    s.vx = s.vx + (a - Ffy * math.sin(delta) / m + s.vy * s.omega) * dt
    s.vy = s.vy + (Fry / m + Ffy * math.cos(delta) / m - s.vx * s.omega) * dt
    s.omega = s.omega + (Ffy * Lf * math.cos(delta) - Fry * Lr) / Iz * dt

    return s


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
    vx, vy = [], []
    time = []
    t = 0.0

    for (ai, di) in zip(a, delta):
        t = t + dt
        state = update(state, ai, di)
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        vx.append(state.vx)
        vy.append(state.vy)
        time.append(t)

    flg, ax = plt.subplots(1)
    plt.plot(x, y)
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.axis("equal")
    plt.grid(True)

    flg, ax = plt.subplots(1)
    plt.plot(time, np.array(vx) * 3.6)
    plt.xlabel("Time[km/h]")
    plt.ylabel("velocity[m]")
    plt.grid(True)

    #  flg, ax = plt.subplots(1)
    #  plt.plot([math.degrees(ibeta) for ibeta in beta])
    #  plt.grid(True)

    plt.show()
