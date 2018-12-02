# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/12/2 16:21
from unittest import TestCase
import numpy as np
from supervised_learning.SupportVectorMachine import RBF


# __file__ = test_RBF.py
class TestRBF(TestCase):
    def test1(self):
        rbf = RBF(1)
        X = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        Y = [1, 2, 3]
        print(rbf(X, Y))

    def test2(self):
        rbf = RBF(1)
        X = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        Y = np.array([[1, 2, 3],
                      [4, 5, 6]])
        print(rbf(X, Y))
