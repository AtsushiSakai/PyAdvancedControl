# -*- coding: utf-8 -*-
u"""
unittest code

author Atsushi Sakai
"""
import unittest
import mpc_tracking.mpc_tracking


class Test(unittest.TestCase):

    def test_1(self):
        mpc_tracking.mpc_tracking.test1()


if __name__ == '__main__':
    unittest.main()
