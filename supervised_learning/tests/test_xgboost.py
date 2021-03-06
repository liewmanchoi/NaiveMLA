# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/12/15 10:21
from unittest import TestCase
from xgboost import sklearn
from supervised_learning.xgboost import XGBClassifier
from supervised_learning.xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston


# __file__ = test_xgboost.py
class TestXGBClassifier(TestCase):
    def test1(self):
        cancer = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target,
                                                            random_state=1)
        my_clf = XGBClassifier(n_estimators=20, reg_lambda=1, gamma=0, max_depth=3, learning_rate=0.1)
        # xgb_clf = sklearn.XGBClassifier(n_estimators=1, max_depth=3)
        my_clf.fit(X_train, y_train)
        # xgb_clf.fit(X_train, y_train)
        print("\nAccuracy on training set: {:.3f}".format(my_clf.score(X_train, y_train)))
        print("Accuracy on test set: {:.3f}".format(my_clf.score(X_test, y_test)))
        # print("Accuracy on training set: {:.3f}".format(xgb_clf.score(X_train, y_train)))
        # print("Accuracy on test set: {:.3f}".format(xgb_clf.score(X_test, y_test)))


class TestXGBRegressor(TestCase):
    def test(self):
        boston = load_boston()
        X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=1)
        my_reg = XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3, learning_rate=0.1)
        xgb_reg = sklearn.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3, learning_rate=0.1)
        my_reg.fit(X_train, y_train)
        xgb_reg.fit(X_train, y_train)
        print(my_reg.score(X_test, y_test))
        print(xgb_reg.score(X_test, y_test))
