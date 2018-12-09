# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/12/9 12:04
from unittest import TestCase
from unsupervised_learning import KMeans
from sklearn.datasets import make_blobs
from sklearn import cluster
import mglearn
import matplotlib.pyplot as plt


# __file__ = test_KMeans.py
class TestKMeans(TestCase):
    def test(self):
        X, y = make_blobs()
        my_kmeans = KMeans(n_clusters=3)
        sk_kmeans = cluster.KMeans(n_clusters=3)
        my_kmeans.fit(X)
        sk_kmeans.fit(X)
        print(my_kmeans.predict(X))
        print(sk_kmeans.predict(X))
        mglearn.discrete_scatter(X[:, 0], X[:, 1], my_kmeans.labels_, markers='o')
        mglearn.discrete_scatter(my_kmeans.cluster_centers_[:, 0], my_kmeans.cluster_centers_[:, 1],
                                 [0, 1, 2], markers="^", markeredgewidth=5)
        plt.show()
