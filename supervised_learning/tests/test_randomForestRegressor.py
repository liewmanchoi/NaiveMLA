# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/11/23 20:47
from unittest import TestCase
from supervised_learning import RandomForestRegressor
import sklearn.ensemble
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression, load_boston


# __file__ = test_randomForestRegressor.py
class TestRandomForestRegressor(TestCase):
    def test(self):
        my_reg = RandomForestRegressor(n_estimators=10, max_depth=5)
        sk_reg = sklearn.ensemble.RandomForestRegressor(n_estimators=10, max_depth=5)
        data = make_regression(n_samples=400, n_features=10)
        X_train, X_test, y_train, y_test = train_test_split(data[0], data[1])
        my_reg.fit(X_train, y_train)
        sk_reg.fit(X_train, y_train)
        print("Accuracy on training set: {:.3f}".format(my_reg.score(X_train, y_train)))
        print("Accuracy on training set: {:.3f}".format(sk_reg.score(X_train, y_train)))

    def test_boston(self):
        my_reg = RandomForestRegressor(n_estimators=10, max_depth=5)
        # sk_reg = sklearn.ensemble.RandomForestRegressor(n_estimators=10, max_depth=5)
        boston = load_boston()
        X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target)
        my_reg.fit(X_train, y_train)
        # sk_reg.fit(X_train, y_train)
        # print("Accuracy on training set: {:.3f}".format(my_reg.score(X_train, y_train)))
        # print("Accuracy on training set: {:.3f}".format(sk_reg.score(X_train, y_train)))