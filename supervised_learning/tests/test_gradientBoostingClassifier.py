# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/11/26 11:47
from unittest import TestCase
import sklearn.ensemble
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from supervised_learning.gradientboosting import GradientBoostingClassifier


# __file__ = test_gradientBoostingClassifier.py
class TestGradientBoostingClassifier(TestCase):
    def test(self):
        cancer = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target,
                                                            random_state=0)
        my_clf = GradientBoostingClassifier(n_estimators=3, learning_rate=0.15)
        sk_clf = sklearn.ensemble.GradientBoostingClassifier(n_estimators=3)
        my_clf.fit(X_train, y_train)
        sk_clf.fit(X_train, y_train)
        print("Accuracy on training set: {:.3f}".format(my_clf.score(X_train, y_train)))
        print("Accuracy on test set: {:.3f}".format(my_clf.score(X_test, y_test)))
        print("Accuracy on training set: {:.3f}".format(sk_clf.score(X_train, y_train)))
        print("Accuracy on test set: {:.3f}".format(sk_clf.score(X_test, y_test)))
