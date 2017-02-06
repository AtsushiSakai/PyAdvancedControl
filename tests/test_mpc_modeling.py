# -*- coding: utf-8 -*-
u"""
unittest code

author Atsushi Sakai
"""
import unittest
import mpc_modeling.mpc_modeling


class Test(unittest.TestCase):

    def test_1(self):
        mpc_modeling.mpc_modeling.test1()

    def test_2(self):
        mpc_modeling.mpc_modeling.test2()

    def test_3(self):
        mpc_modeling.mpc_modeling.test3()

    def test_4(self):
        mpc_modeling.mpc_modeling.test4()

    def test_5(self):
        mpc_modeling.mpc_modeling.test5()

    def test_6(self):
        mpc_modeling.mpc_modeling.test6()


if __name__ == '__main__':
    unittest.main()
