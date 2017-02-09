# -*- coding: utf-8 -*-
u"""
unittest code

author Atsushi Sakai
"""
import unittest
import mpc_modeling.mpc_modeling_with_ECOS


class Test(unittest.TestCase):

    def test_3(self):
        mpc_modeling.mpc_modeling_with_ECOS.test3()

    def test_4(self):
        mpc_modeling.mpc_modeling_with_ECOS.test4()

    def test_5(self):
        mpc_modeling.mpc_modeling_with_ECOS.test5()

    def test_6(self):
        mpc_modeling.mpc_modeling_with_ECOS.test6()


if __name__ == '__main__':
    unittest.main()
