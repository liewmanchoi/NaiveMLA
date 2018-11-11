# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/11/10 21:27
from unittest import TestCase
import numpy as np
from supervised_learning.linear_model import Lasso, Ridge
import mglearn
from sklearn.model_selection import train_test_split
import logging


# __file__ = test_lasso.py
class TestLasso(TestCase):
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.INFO)

    def test(self):
        # X, y = mglearn.datasets.load_extended_boston()
        # X_train, X_test, y_train, y_test = train_test_split(X, y)
        X_train = np.array([[1], [2], [3], [4],[5], [6]])
        y_train = np.array([2, 4, 6, 8, 10, 12])
        ridge = Ridge(alpha=0.1)
        lasso = Lasso(alpha=0.06)
        lasso.fit(X_train, y_train)
        ridge.fit(X_train, y_train)
        print("Lasso training set score: {:.2f}\n".format(lasso.score(X_train, y_train)))
        # print("Lasso test set score: {:.2f}".format(lasso.score(X_test, y_test)))
        print("Ridge training set score: {:.2f}\n".format(ridge.score(X_train, y_train)))
        # print("Ridge test set score: {:.2f}".format(ridge.score(X_test, y_test)))
