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

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.exp(-self._gamma * np.sum(np.square(x1 - x2)))


class SVC(object):
    def __init__(self, C: float = 1.0, kernel: str = "rbf", gamma: float = None, tol: float = 1e-3, max_iter: int = -1):
        self._C: float = C
        self._kernel_name: str = kernel
        self._kerner: Callable = None
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

            self._kerner = RBF(gamma=self._gamma)
        else:
            raise AttributeError("kernel must be RBF")

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
