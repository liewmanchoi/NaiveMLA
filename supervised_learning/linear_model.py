# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/11/10 10:14
# __file__ = linear_model.py

import numpy as np
import abc
import logging
import math
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.metrics import log_loss


class L1Regularizer(object):
    def __call__(self, w: np.ndarray):
        n_features = w.shape
        return np.linalg.norm(w, ord=1) / n_features

    @staticmethod
    def grad(w: np.ndarray):
        n_features = w.shape
        return np.sign(w) / n_features


class L2Regularizer(object):
    def __call__(self, w: np.ndarray):
        n_features = w.shape
        return 0.5 * np.linalg.norm(w, ord=2) / n_features

    @staticmethod
    def grad(w: np.ndarray):
        return np.mean(w)


class L1L2Regularizer(object):
    def __init__(self, alpha=1.0, l1_ratio=0.5):
        self.alpha1 = alpha * l1_ratio
        self.alpha2 = alpha - self.alpha1

    def __call__(self, w: np.ndarray):
        return self.alpha1 * L1Regularizer()(w) + self.alpha2 * L2Regularizer()(w)

    def grad(self, w: np.ndarray):
        return self.alpha1 * L1Regularizer.grad(w) + self.alpha2 * L2Regularizer.grad(w)


class MeanSquaredError(object):
    def __call__(self, X: np.ndarray, w: np.ndarray, y: np.ndarray):
        return mean_squared_error(np.dot(X, w), y)

    def grad(self, X: np.ndarray, w: np.ndarray, y: np.ndarray):
        n_samples = X.shape[0]
        return 1 / n_samples * 2 * np.dot(X.T, np.dot(X, w) - y)


class CrossEntropy(object):
    @staticmethod
    def sigmoid(X: np.ndarray, w: np.ndarray) -> np.ndarray:
        value = 1 / (1 + np.power(1/math.e, np.dot(X, w)))
        return value

    def __call__(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> float:
        y_true = y
        predict_proba_1 = np.array([CrossEntropy.sigmoid(X, w)]).T
        predict_proba_0 = 1 - predict_proba_1
        y_pred = np.concatenate((predict_proba_0, predict_proba_1), axis=1)
        cross_entropy = log_loss(y_true, y_pred)
        logging.debug("cross entropy is {}".format(cross_entropy))
        return cross_entropy

    def grad(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> np.ndarray:
        grad = np.dot(CrossEntropy.sigmoid(X, w) - y, X)
        return grad


class BaseRegression(metaclass=abc.ABCMeta):
    _augmented_coef: np.ndarray
    n_iters_: int

    def __init__(self, alpha: float, l1_ratio: float, penalty: str, max_iter: int):
        logging.info("A {} object has been initialized.".format(self.__class__.__name__))
        self.max_iter = max_iter
        self._set_cost_function()
        self._set_regularizer(alpha, l1_ratio, penalty)
        self._rate = 0.01
        self._tolerance = 0.001
        self.n_iters_ = 0

    def _set_regularizer(self, alpha: float, l1_ratio: float, penalty: str):
        if penalty.lower() == "l1":
            self._regularizer = L1L2Regularizer(alpha, 1.0)
        elif penalty.lower() == "l2":
            self._regularizer = L1L2Regularizer(alpha, 0.0)
        else:
            self._regularizer = L1L2Regularizer(alpha, l1_ratio)

    @abc.abstractmethod
    def _set_cost_function(self):
        self._cost_func = None

    def _init_augmented_coef(self, n_features: int):
        limit = 1 / math.sqrt(1 + n_features)
        self._augmented_coef = np.random.uniform(-1 * limit, limit, (1 + n_features, ))

    def _augment_feature_matrix(self, X: np.ndarray) -> np.ndarray:
        ones = np.ones(shape=[X.shape[0], 1])
        augmented_X = np.concatenate([X, ones], axis=1)
        return augmented_X

    def set_learning_rate(self, rate: float):
        self._rate = rate

    def set_tolerance(self, tolerance: float):
        self._tolerance = tolerance

    def _get_gradient_descent(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> float:
        gradient_descent = self._cost_func.grad(X, w, y) + self._regularizer.grad(w)
        return gradient_descent

    def fit(self, X: np.ndarray, y: np.ndarray):
        logging.info("The training process has begun.")
        n_features = X.shape[1]
        self._init_augmented_coef(n_features)

        X_train = self._augment_feature_matrix(X)
        y_train = y

        errors = [self._cost_func(X_train, self._augmented_coef, y_train) + self._regularizer(self._augmented_coef)]
        for i in range(self.max_iter):
            self._augmented_coef -= self._rate * self._get_gradient_descent(X_train, self._augmented_coef, y_train)
            self.n_iters_ += 1
            loss = self._cost_func(X_train, self._augmented_coef, y_train) + self._regularizer(self._augmented_coef)
            errors.append(loss)
            logging.info("Iteration: {}, loss: {}".format(i, loss))
            if abs(loss - errors[i]) < self._tolerance:
                logging.info("Convergence has reached.")
                break

        if self.n_iters_ == self.max_iter:
            logging.info("Iteration times equals to max_iter.")

        logging.info("coef_ is {}, \n intercept_ is {}".format(self._augmented_coef[:-1], self._augmented_coef[-1]))

    @abc.abstractmethod
    def predict(self, X: np.ndarray):
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return r2_score(y, self.predict(X))


class ElasticNet(BaseRegression):
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000):
        super().__init__(alpha, l1_ratio, "elastic", max_iter)

    def _set_cost_function(self):
        self._cost_func = MeanSquaredError()

    def predict(self, X: np.ndarray):
        X_test = self._augment_feature_matrix(X)
        return np.dot(X_test, self._augmented_coef)


class Lasso(ElasticNet):
    def __init__(self, alpha=1.0, max_iter=1000):
        super().__init__(alpha, 1.0, max_iter)


class Ridge(ElasticNet):
    def __init__(self, alpha=1.0, max_iter=1000):
        super().__init__(alpha, 0, max_iter)


class LogisticRegression(BaseRegression):
    def __init__(self, penalty="l2", C=0, max_iter=100):
        super().__init__(alpha=C, l1_ratio=C, penalty=penalty, max_iter=max_iter)

    def _set_cost_function(self):
        self._cost_func = CrossEntropy()

    def predict(self, X: np.ndarray):
        X_test = self._augment_feature_matrix(X)
        predict_proba = CrossEntropy.sigmoid(X_test, self._augmented_coef)
        y_pred = (predict_proba > 0.5).astype(int, copy=False)
        return y_pred

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))
