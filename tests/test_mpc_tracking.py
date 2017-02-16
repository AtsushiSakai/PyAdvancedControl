# -*- coding: utf-8 -*-
u"""
unittest code

author Atsushi Sakai
"""
import unittest

import sys
sys.path.append("./mpc_tracking/")

import mpc_tracking.mpc_tracking


class Test(unittest.TestCase):

    def test_1(self):
        mpc_tracking.mpc_tracking.test1()

    def test_2(self):
        mpc_tracking.mpc_tracking.test2()

    def test_3(self):
        mpc_tracking.mpc_tracking.test3()

    def test_4(self):
        mpc_tracking.mpc_tracking.test4()

    def test_5(self):
        mpc_tracking.mpc_tracking.test5()


if __name__ == '__main__':
    unittest.main()
