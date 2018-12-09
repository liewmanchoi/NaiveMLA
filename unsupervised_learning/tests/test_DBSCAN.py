# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/12/9 16:43
from unittest import TestCase
from unsupervised_learning import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# __file__ = test_DBSCAN.py
class TestDBSCAN(TestCase):
    def test(self):
        X, y = make_moons(n_samples=200, noise=0.05)
        my_dbscan = DBSCAN(eps=0.2)
        cluster = my_dbscan.fit_predict(X, y)
        print(cluster)
        plt.scatter(X[:, 0], X[:, 1], c=cluster, s=60)
        plt.show()
