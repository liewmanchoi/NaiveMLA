# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/12/10 21:02
from unittest import TestCase
from unsupervised_learning import PCA
from sklearn import decomposition
from sklearn.datasets import load_breast_cancer


# __file__ = test_PCA.py
class TestPCA(TestCase):
    def test(self):
        cancer = load_breast_cancer()
        my_pca = PCA(n_components=2)
        sk_pca = decomposition.PCA(n_components=2)
        print(my_pca.fit_transform(cancer.data))
        print(sk_pca.fit_transform(cancer.data))
