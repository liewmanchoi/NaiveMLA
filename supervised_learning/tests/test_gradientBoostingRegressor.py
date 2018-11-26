# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/11/26 19:12
from unittest import TestCase
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression, load_boston
import sklearn.ensemble
from supervised_learning.GradientBoosting import GradientBoostingRegressor

# __file__ = test_gradientBoostingRegressor.py
class TestGradientBoostingRegressor(TestCase):
    def test(self):
        boston = load_boston()
        X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=1)

        my_reg = GradientBoostingRegressor(max_depth=3)
        sk_reg = sklearn.ensemble.GradientBoostingRegressor()
        my_reg.fit(X_train, y_train)
        sk_reg.fit(X_train, y_train)

        print("Accuracy on training set: {:.3f}".format(my_reg.score(X_train, y_train)))
        print("Accuracy on test set: {:.3f}".format(my_reg.score(X_test, y_test)))
        print("Accuracy on training set: {:.3f}".format(sk_reg.score(X_train, y_train)))
        print("Accuracy on test set: {:.3f}".format(sk_reg.score(X_test, y_test)))

