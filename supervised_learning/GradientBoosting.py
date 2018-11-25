# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/11/22 16:47
# __file__ = GradientBoosting.py


import numpy as np
import abc
from typing import List
from supervised_learning.DecisionTree import BaseDecisionTree
from supervised_learning import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.special import expit


class MeanEstimator(object):
    # for regression
    def __init__(self):
        self.mean: float = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.mean = float(np.mean(y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = np.empty(shape=X.shape[0], dtype=np.float64)
        y_pred.fill(self.mean)
        return y_pred


class LogOddsEstimator(object):
    # for binary classification
    def __init__(self):
        self.prior: float = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pos = np.sum(y)
        neg = y.size - pos

        if neg == 0 or pos == 0:
            raise ValueError('y contains non binary labels.')

        # p = 1 / (1 + exp(-f(x)) -> f(x) = log(p / (1 - p))
        self.prior = float(np.log(pos / neg))

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = np.empty(shape=X.shape[0], dtype=np.float64)
        y_pred.fill(self.prior)
        return y_pred


class LossFunction(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def init_estimator(self):
        # init estimator for loss function
        pass

    @abc.abstractmethod
    def __call__(self, y_true, y_pred):
        # compute the loss
        pass

    @abc.abstractmethod
    def negative_gradient(self, y_true, y_pred):
        # compute the negative gradient
        pass


class LeastSquareError(LossFunction):
    def init_estimator(self):
        return MeanEstimator()

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.square(y_true - y_pred)))

    def negative_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_true - y_pred


class BinomialDeviance(LossFunction):
    def init_estimator(self):
        return LogOddsEstimator()

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # L(y_true, y_pred) = y_true * log(1 + exp(-1 * y_pred)) + (1 - y_true) * log(1 + exp(y_pred))
        # y_true - {0, 1}
        # P(y=1 | x) = 1 / (1 + exp(-y_pred)) = expit(y_pred)
        # np.logaddexp(x1, x2) = np.log(np.exp(x1) + np.exp(x2))
        return float(np.mean(np.multiply(y_true, np.logaddexp(0.0, -y_pred)) +
                       np.multiply(1 - y_true, np.logaddexp(0.0, y_pred))))

    def negative_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # -dL/dy_pred = y_true - 1 / (1 + exp(f(x)) = y_true + 1 - 1 / (1 + exp(-f(x))
        # expit(x) = 1 / (1 + exp(-x)
        return y_true - expit(-y_pred)


class BaseGradientBoosting(object):
    def __init__(self, learning_rate: float, n_estimators: int, max_depth: int, min_samples_split,
                 min_impurity_split: float, max_features: int, early_stop: bool,
                 validation_fraction: float, tol: float):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self._actual_n_estimators: int = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.max_features = max_features
        self.early_stop = early_stop
        self.validation_fraction = validation_fraction
        self.tol = tol
        self._estimators: List['BaseDecisionTree'] = list()

    @property
    def n_estimators_(self):
        return self._actual_n_estimators

    @property
    def estimators_(self) -> List['BaseDecisionTree']:
        return self._estimators

