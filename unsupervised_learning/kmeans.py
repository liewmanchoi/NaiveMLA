# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/12/9 10:05
# __file__ = kmeans.py

import numpy as np


class KMeans(object):
    def __init__(self, n_clusters: int = 8, max_iter: int = 300):
        self._n_clusters: int = n_clusters
        self._max_iter: int = max_iter
        self._cluster_centers: np.ndarray = None
        self._labels: np.ndarray = None

    @property
    def cluster_centers_(self) -> np.ndarray:
        return self._cluster_centers

    @property
    def labels_(self) -> np.ndarray:
        return self._labels

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "KMeans":
        mask = np.random.choice(X.shape[0], self._n_clusters, replace=False)
        self._cluster_centers = X[mask]

        iters = 0
        while iters < self._max_iter:
            iters += 1
            prev_cluster_centers = self._cluster_centers

            self._get_labels(X)
            self._update_cluster_centers(X)

            diff = np.sum(prev_cluster_centers - self._cluster_centers)
            if diff == 0:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        labels = np.argmin(self._update_distance(X), axis=1)
        return labels

    def fit_predict(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        self.fit(X)
        return self._labels

    def _update_distance(self, X: np.ndarray) -> np.ndarray:
        distance_to_centers = np.ndarray(shape=(X.shape[0], self._n_clusters))
        for idx, cluster_center in enumerate(self._cluster_centers):
            distance_array = np.sum(np.square(X - cluster_center), axis=1)
            distance_to_centers[:, idx] = distance_array

        return distance_to_centers

    def _get_labels(self, X: np.ndarray) -> None:
        self._labels = np.argmin(self._update_distance(X), axis=1)

    def _update_cluster_centers(self, X: np.ndarray) -> None:
        for idx in range(self._n_clusters):
            mask = self._labels == idx
            self._cluster_centers[idx] = np.mean(X[mask], axis=0)
