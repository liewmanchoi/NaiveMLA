# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/12/9 18:12
# __file__ = pca.py

import numpy as np


class PCA(object):
    def __init__(self, n_components: int):
        self._n_components: int = n_components
        self._mean: np.ndarray = None
        self._components: np.ndarray = None

    def fit(self, X: np.ndarray, y=None) -> "PCA":
        X = np.atleast_2d(X.copy())
        n_samples, n_features = X.shape
        if n_samples < self._n_components or n_features < self._n_components:
            self._n_components = min(n_samples, n_features) - 1

        self._mean = np.mean(X, axis=0)
        # 中心化数组
        X -= self._mean

        # 使用SVD算法求解
        _, S, V = np.linalg.svd(X, full_matrices=False)
        # 设置基向量矩阵
        self._components = V[:self._n_components]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X -= self._mean
        X_transformed = np.dot(X, self._components.T)
        return X_transformed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X_transformed = self.fit(X).transform(X)
        return X_transformed

    @property
    def mean_(self) -> np.ndarray:
        return self._mean

    @property
    def n_components_(self) -> int:
        return self._n_components

    @property
    def components_(self) -> np.ndarray:
        return self._components
