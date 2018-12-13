# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/11/22 16:47
# __file__ = gradientboosting.py


import numpy as np
import math
import abc
from typing import List
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
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

    @staticmethod
    def output_to_proba(y_pred):
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

    @staticmethod
    def output_to_proba(y_pred: np.ndarray) -> np.ndarray:
        proba = expit(y_pred)
        return proba


class BaseGradientBoosting(object):
    def __init__(self, loss: 'LossFunction', learning_rate: float, n_estimators: int, max_depth: int,
                 min_samples_split: int, min_impurity_split: float, max_features: int):
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.max_features = max_features
        self._n_features: int = None
        self._estimators: List['DecisionTreeRegressor'] = list()
        self.init_: None

    def _init_state(self) -> None:
        # init model & self._estimators
        self.init_ = self.loss.init_estimator()

        for _ in range(self.n_estimators):
            self._estimators.append(
                DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                      min_impurity_split=self.min_impurity_split, max_features=self.max_features)
            )

    def _init_decision_function(self, X: np.ndarray) -> np.ndarray:
        score = self.init_.predict(X).astype(np.float64)
        return score

    @property
    def n_estimators_(self):
        return self.n_estimators

    @property
    def estimators_(self) -> List['DecisionTreeRegressor']:
        return self._estimators

    @abc.abstractmethod
    def _check_max_features(self) -> None:
        # invoke after self._n_features is decided
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, self._n_features = X.shape
        self._check_max_features()

        # init model & self._estimators
        self._init_state()
        self.init_.fit(X, y)
        # 赋予初值
        f = self._init_decision_function(X)

        # 逐个拟合负梯度值
        for estimator in self._estimators:
            # 计算负梯度值
            residual = self.loss.negative_gradient(y_true=y, y_pred=f)
            estimator.fit(X, residual)
            # 注意shrinkage机制
            f += self.learning_rate * estimator.predict(X)

        return self

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        # 回归器可以直接使用本函数，分类器还需要转换为概率和类别
        f = self._init_decision_function(X)

        for estimator in self._estimators:
            f += self.learning_rate * estimator.predict(X)

        return f

    @abc.abstractmethod
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        pass


class GradientBoostingClassifier(BaseGradientBoosting):
    def __init__(self, learning_rate: float =0.1, n_estimators: int =100,
                 max_depth: int =3, min_samples_split: int =2, min_impurity_split: float =0.0,
                 max_features: int = None):
        super().__init__(BinomialDeviance(), learning_rate, n_estimators, max_depth, min_samples_split,
                         min_impurity_split, max_features)

    def _check_max_features(self) -> None:
        if self.max_features is None:
            self.max_features = int(math.sqrt(self._n_features))

    def predict(self, X: np.ndarray) -> np.ndarray:
        f = super().predict(X)
        proba = np.empty(shape=(X.shape[0], 2))
        proba[:, 1] = self.loss.output_to_proba(f)
        proba[:, 0] = 1 - proba[:, 1]
        y_pred = np.argmax(proba, axis=1)

        return y_pred

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))


class GradientBoostingRegressor(BaseGradientBoosting):
    def __init__(self, learning_rate: float =0.1, n_estimators: int =100,
                 max_depth: int =3, min_samples_split: int =2, min_impurity_split: float =0.0,
                 max_features: int = None):
        super().__init__(LeastSquareError(), learning_rate, n_estimators, max_depth, min_samples_split,
                         min_impurity_split, max_features)

    def _check_max_features(self) -> None:
        if self.max_features is None:
            self.max_features = self._n_features

    def predict(self, X: np.ndarray) -> np.ndarray:
        return super().predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return r2_score(y, self.predict(X))
