# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/12/1 20:44
# __file__ = SupportVectorMachine.py

import numpy as np
from typing import Callable


class RBF(object):
    def __init__(self, gamma: float):
        self._gamma = gamma

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        # shape: (X.shape[0], Y.shape[0])
        Y = np.atleast_2d(Y)
        kernel_matrix = np.empty(shape=(X.shape[0], Y.shape[0]))
        for idx, y in enumerate(Y):
            kernel_matrix[:, idx] = np.exp(-self._gamma * np.sum(np.square(X - y), axis=1))

        return kernel_matrix


class SVC(object):
    def __init__(self, C: float = 1.0, kernel: str = "rbf", gamma: float = None, tol: float = 1e-3, max_iter: int = -1):
        self._C: float = C
        self._kernel_name: str = kernel
        self._kernel: Callable = None
        self._gamma: float = gamma
        self._tol: float = tol
        self._max_iter: int = max_iter
        self._support: np.ndarray = None  # indices of support vectors
        self._support_vectors: np.ndarray = None  # support vectors
        self._dual_coef: np.ndarray = None  # coefficients of the support vectors in decision function
        self._intercept: float = None  # constant in decision function
        self._n_features: int = None  # number of features

    def _init_kernel(self) -> None:
        if self._kernel_name == "rbf":
            if self._gamma is None:
                self._gamma = 1 / self._n_features

            self._kernel = RBF(gamma=self._gamma)
        else:
            raise AttributeError("kernel must be RBF")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVC":
        n_samples, self._n_features = X.shape
        self._init_kernel()

        self._optimize_by_SMO(X, y)
        return self

    def _optimize_by_SMO(self, X: np.ndarray, y: np.ndarray):
        # min 1/2 alpha.T * Q * alpha - np.sum(alpha)
        # Q[i][j] = y[i]*y[j]*K[i][j]
        K_matrix = self._kernel(X, X)

        alpha = np.empty(shape=X.shape[0])


    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        K_matrix = self._kernel(X, self._support_vectors)

        # dual_coef[i] = alpha[i] * y[i]
        y_pred = np.sign(np.dot(K_matrix, self._dual_coef) + self._intercept)
        return y_pred

    # indices of support vectors
    @property
    def support_(self) -> np.ndarray:
        return self._support

    # support vectos
    @property
    def support_vectors_(self) -> np.ndarray:
        return self._support_vectors

    # coefficients of the support vectors in decision function
    @property
    def dual_coef_(self) -> np.ndarray:
        return self._dual_coef

    # constant in decision function
    @property
    def intercept_(self) -> float:
        return self._intercept
