# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/12/9 13:56
# __file__ = DBSCAN.py

import numpy as np
from typing import List
from collections import deque


class DBSCAN(object):
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self._eps = eps
        self._min_samples = min_samples
        self._core_sample_indices: List[int] = list()
        self._labels: np.ndarray = None
        self._components: np.ndarray = None

    @property
    def core_sample_indices_(self):
        return np.array(self._core_sample_indices)

    @property
    def labels_(self):
        return self._labels

    @property
    def components_(self) -> np.ndarray:
        return self._components

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "DBSCAN":
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape
        self._labels = np.empty(shape=n_samples)

        # 获取核心对象集合
        self._get_core_samples(X)
        # 核心对象特征向量集合
        self._components = X[self.core_sample_indices_]

        unvisited_samples_set = set(range(n_samples))
        core_samples_set = set(self._core_sample_indices)
        label = 0

        while len(core_samples_set) != 0:
            sample_to_visit = core_samples_set.pop()
            queue = deque()
            unvisited_samples_set.remove(sample_to_visit)
            cluster = set()

            while len(queue) != 0:
                idx = queue.pop()
                neighbors = set(self._get_neighbors(X, idx))
                cluster.add(idx)

                if idx in self._core_sample_indices:
                    samples = neighbors.intersection(unvisited_samples_set)
                    queue.extend(samples)
                    unvisited_samples_set.difference_update(samples)
                    cluster.update(samples)

            core_samples_set.difference_update(cluster)
            self._labels[cluster] = label
            label += 1

        return self

    def fit_predict(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        return self.fit(X, y).labels_

    def _get_neighbors(self, X: np.ndarray, idx: int) -> List[int]:
        distances: np.ndarray = np.sqrt(np.sum(np.square(X - X[idx]), axis=1))
        neighbors_list = np.argwhere(distances <= self._eps).ravel().tolist()
        return neighbors_list

    def _get_core_samples(self, X: np.ndarray) -> None:
        for idx, _ in enumerate(X):
            if len(self._get_neighbors(X, idx)) > self._min_samples:
                self._core_sample_indices.append(idx)
