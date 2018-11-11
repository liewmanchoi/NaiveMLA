# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/11/11 15:37
from unittest import TestCase
from supervised_learning import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import logging


# __file__ = test_logisticRegression.py
class TestLogisticRegression(TestCase):
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        level=logging.DEBUG)

    def test(self):
        cancer = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target)
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))

