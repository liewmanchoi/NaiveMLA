# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/11/8 11:16
# __file__ = KNearestNeighbor.py


import numpy as np
from scipy.spatial.distance import euclidean
from typing import Callable
from sklearn.metrics import accuracy_score


class KNN(object):
    n_neighbors: int
    distance_metric: Callable
    _X: np.ndarray
    _y: np.ndarray

    def __init__(self, n_neighbors=5, distance_metric=euclidean) -> None:
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'KNN':
        self._X = X_train
        self._y = y_train
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_predict = np.empty(shape=X.shape[0])
        for idx, x_predict in enumerate(X):
            k_nearest_neighbors = np.argsort([self.distance_metric(x_predict, _x) for _x in self._X])[:self.n_neighbors]
            predict = np.argmax(np.bincount(self._y[k_nearest_neighbors]))
            y_predict[idx] = predict

        return y_predict

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
