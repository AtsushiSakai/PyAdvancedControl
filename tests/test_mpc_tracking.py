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

    def test_6(self):
        mpc_tracking.mpc_tracking.test6()

    def test_7(self):
        mpc_tracking.mpc_tracking.test7()

    def test_8(self):
        mpc_tracking.mpc_tracking.test8()

    def test_9(self):
        mpc_tracking.mpc_tracking.test9()

    def test_10(self):
        mpc_tracking.mpc_tracking.test10()

    def test_11(self):
        mpc_tracking.mpc_tracking.test11()

    def test_12(self):
        mpc_tracking.mpc_tracking.test12()


if __name__ == '__main__':
    unittest.main()
