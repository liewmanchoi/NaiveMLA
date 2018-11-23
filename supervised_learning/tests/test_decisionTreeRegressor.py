# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/11/23 11:06
from unittest import TestCase
from supervised_learning import DecisionTreeRegressor
import sklearn.tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression, load_boston


# __file__ = test_decisionTreeRegressor.py
class TestDecisionTreeRegressor(TestCase):
    def test(self):
        my_reg = DecisionTreeRegressor()
        sk_reg = sklearn.tree.DecisionTreeRegressor()
        data = make_regression(n_samples=400, n_features=40)
        X_train, X_test, y_train, y_test = train_test_split(data[0], data[1])
        my_reg.fit(X_train, y_train)
        sk_reg.fit(X_train, y_train)
        print("Accuracy on training set: {:.3f}".format(my_reg.score(X_train, y_train)))
        print("Accuracy on training set: {:.3f}".format(sk_reg.score(X_train, y_train)))

    def test_boston(self):
        my_reg = DecisionTreeRegressor()
        sk_reg = sklearn.tree.DecisionTreeRegressor()
        boston = load_boston()
        X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target)
        my_reg.fit(X_train, y_train)
        sk_reg.fit(X_train, y_train)
        print("Accuracy on training set: {:.3f}".format(my_reg.score(X_train, y_train)))
        print("Accuracy on training set: {:.3f}".format(sk_reg.score(X_train, y_train)))



