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

