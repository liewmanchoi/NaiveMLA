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


class SMOOptimizer(object):
    def __init__(self, C: float, kernel: Callable, tol: float, max_iter: int):
        self._C: float = C
        self._kernel: Callable = kernel
        self._tol: float = tol
        self._max_iter: int = max_iter
        self._alphas: np.ndarray = None
        self._alpha1: float = None
        self._first_idx: int = None
        self._alpha2: float = None
        self._second_idx: int = None
        self._b: float = None
        self._K_matrix: np.ndarray = None
        self._predictions: np.ndarray = None
        self._errors: np.ndarray = None
        self._X: np.ndarray = None
        self._y: np.ndarray = None

    # 更新g(x)的值
    def _update_predictions(self) -> None:
        self._predictions = np.dot(self._K_matrix, np.multiply(self._alphas, self._y)) + self._b

    # 更新误差值: g(x) - y
    def _update_erros(self) -> None:
        self._erros = self._predictions - self._y

    # 外层循环：选择合适的alpha1值
    def _outer_loop(self) -> None:
        for i, alpha in enumerate(self._alphas):
            distance = self._predictions[i] * self._y[i]
            if (0 < alpha < self._C and not (distance == 1)) or \
                    (alpha == 0 and not (distance >= 1)) or \
                    (alpha == self._C and not (distance <= 1)):
                self._alpha1 = alpha
                self._first_idx = i
                return

        self._alpha1 = None
        self._first_idx = None

    # 内层循环：选择合适的alpha2值
    def _inner_loop(self) -> None:
        alpha2 = np.abs(self._errors - self._errors[self._first_idx]).max()
        second_idx = np.argmax(np.abs(self._errors - self._errors[self._first_idx]))

        while second_idx == self._first_idx:
            second_idx = np.random.randint(0, self._errors.size)
            alpha2 = self._alphas[second_idx]

        self._alpha2 = alpha2
        self._second_idx = second_idx

    def optimize(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._X = X
        self._y = y
        self._K_matrix = self._kernel(X, X)
        self._alphas = np.zeros(shape=y.size)
        self._b = 0

        # 计算函数的预测值g(x)
        self._predictions = np.dot(self._K_matrix, np.multiply(self._alphas, self._y)) + self._b
        # 计算误差值
        self._errors = self._predictions - self._y

        iters = 0
        while iters < self._max_iter or self._max_iter < 0:
            iters += 1

            # 外层循环更新alpha1值
            self._outer_loop()

            # 所有alpha值全部符合KKT条件，停止最优化
            if self._alpha1 is None:
                break

            # 内层循环更新alpha2值
            self._inner_loop()

            K11 = self._K_matrix[self._first_idx, self._first_idx]
            K12 = self._K_matrix[self._first_idx, self._second_idx]
            K21 = self._K_matrix[self._second_idx, self._first_idx]
            K22 = self._K_matrix[self._second_idx, self._second_idx]
            y1 = self._y[self._first_idx]
            e1 = self._errors[self._first_idx]
            y2 = self._y[self._second_idx]
            e2 = self._errors[self._second_idx]
            eta = K11 + K22 - 2 * K12

            # 计算新的未经剪辑的alpha2值
            alpha2_unc = self._alpha2 + y2 * (e1 - e2) / eta
            L, H = self._found_bounds()

            # 更新alpha2和alpha1的值
            self._alphas[self._second_idx] = self._clip(L, H, alpha2_unc)
            self._alphas[self._first_idx] = self._alpha1 + y1 * y2 * (self._alpha2 - self._alphas[self._second_idx])

            # 更新b的值
            b1 = -e1 - y1 * K11 * (self._alphas[self._first_idx] - self._alpha1) \
                 - y2 * K21 * (self._alphas[self._second_idx] - self._alpha2) + self._b
            b2 = -e2 - y1 * K12 * (self._alphas[self._first_idx] - self._alpha1) \
                 - y2 * K22 * (self._alphas[self._second_idx] - self._alpha2) + self._b

            if 0 < self._alphas[self._first_idx] < self._C:
                self._b = b1
            elif 0 < self._alphas[self._second_idx] < self._C:
                self._b = b2
            else:
                self._b = (b1 + b1) / 2

            # 更新prediction和e值
            self._predictions[self._first_idx] = \
                np.dot(self._K_matrix[self._first_idx], np.multiply(self._alphas, self._y)) + self._b
            self._predictions[self._second_idx] = \
                np.dot(self._K_matrix[self._second_idx], np.multiply(self._alphas, self._y)) + self._b
            self._errors[self._first_idx] = self._predictions[self._first_idx] - y1
            self._errors[self._second_idx] = self._predictions[self._second_idx] - y2

            # 阈值门槛
            if (self._alphas[self._first_idx] - self._alpha1) ** 2 + (self._alphas[self._second_idx] - self._alpha2) \
                    ** 2 < self._tol:
                break

        return self._alphas

    @staticmethod
    def _clip(L: float, H: float, alpha2_unc: float) -> float:
        if alpha2_unc > H:
            alpha2 = H
        elif alpha2_unc < L:
            alpha2 = L
        else:
            alpha2 = alpha2_unc

        return alpha2

    # 计算alpha2所在端点的界
    def _found_bounds(self) -> tuple:
        alpha1 = self._alpha1
        alpha2 = self._alpha2

        if self._y[self._first_idx] != self._y[self._second_idx]:
            L = max(0.0, alpha2 - alpha1)
            H = min(self._C, self._C + alpha2 - alpha1)
        else:
            L = max(0.0, alpha2 + alpha1 - self._C)
            H = min(self._C, alpha2 + alpha1)

        return L, H


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
        self._optimizer: "SMOOptimizer" = None

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

        self._optimizer = SMOOptimizer(C=self._C, kernel=self._kernel, tol=self._tol, max_iter=self._max_iter)
        alphas = self._optimizer.optimize(X, y)

        self._support = np.nonzero(alphas)
        self._support_vectors = X[self._support]
        self._dual_coef = alphas[self._support]
        return self

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
