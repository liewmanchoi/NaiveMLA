# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/11/8 15:30
from unittest import TestCase
from unittest import TestCase
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from supervised_learning import KNN


# __file__ = test_KNN.py
class TestKNN(TestCase):
    def test_fit(self):
        iris_dataset = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'])
        sk_knn = KNeighborsClassifier()
        knn = KNN()
        sk_knn.fit(X_train, y_train)
        knn.fit(X_train, y_train)
        print("sklearn score: {}\n".format(sk_knn.score(X_test, y_test)))
        print("knn score: {}\n".format(knn.score(X_test, y_test)))


    def test_predict(self):
        self.fail()

    def test_score(self):
        self.fail()
