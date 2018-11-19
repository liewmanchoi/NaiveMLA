# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/11/19 16:03
from unittest import TestCase
from supervised_learning import DecisionTreeClassifier
from sklearn.datasets import load_iris
import sklearn.tree
from sklearn.model_selection import train_test_split


# __file__ = test_decisionTreeClassifier.py
class TestDecisionTreeClassifier(TestCase):
    def test(self):
        cancer = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target)

        my_tree = DecisionTreeClassifier(max_depth=5)
        sk_tree = sklearn.tree.DecisionTreeClassifier(max_depth=5)
        my_tree.fit(X_train, y_train)
        sk_tree.fit(X_train, y_train)
        print("Accuracy on training set: {:.3f}".format(my_tree.score(X_train, y_train)))
        print("Accuracy on training set: {:.3f}".format(sk_tree.score(X_train, y_train)))
