# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/11/12 14:48
from unittest import TestCase
from supervised_learning.NaiveBayes import GaussianNB
import numpy as np
from sklearn import datasets


# __file__ = test_gaussianNB.py
class TestGaussianNB(TestCase):
    def test_fit(self):
        iris = datasets.load_iris()
        clf = GaussianNB()
        y_pred = clf.fit(iris.data, iris.target).predict(iris.data)
        print("Number of mislabeled points out of a total %d points : %d"
              % (iris.data.shape[0], (iris.target != y_pred).sum()))

    def test_predict(self):
        self.fail()

    def test__calculate_log_likehood_matrix(self):
        pass

    def test__classify(self):
        self.fail()
