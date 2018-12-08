# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/11/22 15:21
from unittest import TestCase
from supervised_learning.RandomForest import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import sklearn.ensemble
from sklearn.model_selection import train_test_split
import numpy as np


# __file__ = test_randomForestClassifier.py
class TestRandomForestClassifier(TestCase):
    def test(self):
        cancer = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target)

        my_tree = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=7)
        my_tree.fit(X_train, y_train)
        sk_tree = sklearn.ensemble.RandomForestClassifier(max_depth=5, n_estimators=10)
        sk_tree.fit(X_train, y_train)
        print("Accuracy on training set: {:.3f}".format(my_tree.score(X_train, y_train)))
        print("Accuracy on test set: {:.3f}".format(my_tree.score(X_test, y_test)))
        print("Accuracy on training set: {:.3f}".format(sk_tree.score(X_train, y_train)))
        print("Accuracy on test set: {:.3f}".format(sk_tree.score(X_test, y_test)))
