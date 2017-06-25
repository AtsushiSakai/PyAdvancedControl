#! /usr/bin/python
"""

Inverted Pendulum MPC control

author: Atsushi Sakai

"""

import matplotlib.pyplot as plt
import numpy as np
import math

l_bar = 2.0  # length of bar


def flatten(a):
    return np.array(a).flatten()


def show_cart(xt, theta):
    cart_w = 1.0
    cart_h = 0.5
    radius = 0.1

    cx = np.matrix([-cart_w / 2.0, cart_w / 2.0, cart_w /
                    2.0, -cart_w / 2.0, -cart_w / 2.0])
    cy = np.matrix([0.0, 0.0, cart_h, cart_h, 0.0])
    cy += radius * 2.0

    cx = cx + xt

    bx = np.matrix([0.0, l_bar * math.sin(theta)])
    bx += xt
    by = np.matrix([cart_h, l_bar * math.cos(theta) + cart_h])
    by += radius * 2.0

    angles = np.arange(0.0, math.pi * 2.0, math.radians(3.0))
    ox = [radius * math.cos(a) for a in angles]
    oy = [radius * math.sin(a) for a in angles]

    rwx = np.copy(ox) + cart_w / 4.0 + xt
    rwy = np.copy(oy) + radius
    lwx = np.copy(ox) - cart_w / 4.0 + xt
    lwy = np.copy(oy) + radius

    wx = np.copy(ox) + float(bx[0, -1])
    wy = np.copy(oy) + float(by[0, -1])

    plt.plot(flatten(cx), flatten(cy), "-b")
    plt.plot(flatten(bx), flatten(by), "-k")
    plt.plot(flatten(rwx), flatten(rwy), "-k")
    plt.plot(flatten(lwx), flatten(lwy), "-k")
    plt.plot(flatten(wx), flatten(wy), "-k")
    plt.title("x:" + str(round(xt, 2)) + ",theta:" +
              str(round(math.degrees(theta), 2)))

    plt.axis("equal")


def main():
    pass


def visualize_test():

    #  x = 1.0
    #  theta = math.radians(10.0)
    #  show_cart(x, theta)
    #  plt.show()

    angles = np.arange(-math.pi / 2.0, math.pi / 2.0, math.radians(1.0))

    xl = [2.0 * math.cos(i) for i in angles]

    for x, theta in zip(xl, angles):
        plt.clf()
        show_cart(x, theta)
        plt.pause(0.001)


if __name__ == '__main__':
    #  main()
    visualize_test()
