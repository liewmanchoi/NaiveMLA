# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/11/22 10:37
# __file__ = RandomForest.py

import numpy as np
import abc
import math
from typing import List
from supervised_learning.DecisionTree import BaseDecisionTree
from supervised_learning import DecisionTreeClassifier
from supervised_learning import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score


class BaseRandomForest(object):
    def __init__(self, n_estimators: int, max_depth: int, min_samples_split: int,
                 min_impurity_split: float, max_features: int):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.max_features = max_features
        self._n_features: int = None
        self._estimators: List['BaseDecisionTree'] = list()

    @property
    def n_features_(self) -> int:
        return self._n_features

    @property
    def estimators_(self) -> List['BaseDecisionTree']:
        return self._estimators

    @abc.abstractmethod
    def _init_estimators(self) -> None:
        pass

    @abc.abstractmethod
    def _check_max_features(self) -> None:
        # invoke after self._n_features is decided
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseRandomForest":
        n_samples, self._n_features = X.shape
        self._check_max_features()
        self._init_estimators()

        for estimator in self.estimators_:
            bootstrap_idx = np.random.choice(np.arange(n_samples), n_samples, replace=True)

            # generate subset
            X_subset = X[bootstrap_idx]
            y_subset = y[bootstrap_idx]
            # fit each estimator
            estimator.fit(X_subset, y_subset)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_preds = np.empty(shape=(X.shape[0], self.n_estimators), dtype=np.int64)

        for idx, estimator in enumerate(self._estimators):
            y_preds[:, idx] = estimator.predict(X)

        y_pred = np.empty(shape=X.shape[0])
        for idx, values in enumerate(y_preds):
            y_pred[idx] = self._get_ensemble_value(values)

        return y_pred

    @abc.abstractmethod
    def _get_ensemble_value(self, values: np.ndarray):
        pass


class RandomForestClassifier(BaseRandomForest):
    def __init__(self, n_estimators: int=100, max_depth: int=float("inf"), min_samples_split: int = 2,
                 min_impurity_split: float=0, max_features: int=None):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                         min_impurity_split=min_impurity_split, max_features=max_features)
        self._classes: np.ndarray = None
        self._n_classes: int = None

    @property
    def classes_(self) -> np.ndarray:
        return self._classes

    @property
    def n_classes_(self) -> int:
        return self._n_classes

    def _init_estimators(self):
        for _ in range(self.n_estimators):
            self._estimators.append(
                DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                       min_impurity_split=self.min_impurity_split, max_features=self.max_features))

    def _check_max_features(self) -> None:
        if self.max_features is None:
            self.max_features = int(math.sqrt(self._n_features))

    def _get_ensemble_value(self, values: np.ndarray) -> int:
        return int(np.argmax(np.bincount(values)))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))


class RandomForestRegressor(BaseRandomForest):
    def __init__(self, n_estimators: int=100, max_depth: int=float("inf"), min_samples_split: int = 2,
                 min_impurity_split: float=0, max_features: int=None):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                         min_impurity_split=min_impurity_split, max_features=max_features)

    def _init_estimators(self):
        for _ in range(self.n_estimators):
            self._estimators.append(
                DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                      min_impurity_split=self.min_impurity_split, max_features=self.max_features))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressor":
        if self.max_features is None:
            self.max_features = int(X.shape[1])

        super().fit(X, y)
        return self

    def _get_ensemble_value(self, values: np.ndarray) -> float:
        return float(np.mean(values))

    def _check_max_features(self) -> None:
        if self.max_features is None:
            self.max_features = self._n_features

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return r2_score(y, self.predict(X))
