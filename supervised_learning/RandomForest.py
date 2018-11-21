# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/11/22 10:37
# __file__ = RandomForest.py

import numpy as np
import abc
from typing import List
from supervised_learning.DecisionTree import BaseDecisionTree


class BaseRandomForest(object):
    def __init__(self, n_estimators:int, max_depth:int, min_samples_split,
                 min_impurity_split: float, max_features: int):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.max_features = max_features
        self._n_features: int = None
        self._estimators: List['BaseDecisionTree'] = list()
        self._in_bag_features: List[np.ndarray] = list()

    @property
    def n_features_(self) -> int:
        return self._n_features

    @property
    def estimators_(self) -> List['BaseDecisionTree']:
        return self._estimators

    @abc.abstractmethod
    def _init_estimators(self) -> None:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseRandomForest':
        self._init_estimators()
        n_samples, self._n_features = X.shape

        for estimator in self.estimators_:
            bootstrap_idx = np.random.choice(np.arange(n_samples), n_samples, replace=True)
            # random feature selection
            feature_idx = np.random.choice(np.arange(self.n_features_), self.max_features, replace=False)
            self._in_bag_features.append(feature_idx)

            # generate subset
            X_subset = X[np.ix_(bootstrap_idx, feature_idx)]
            y_subset = y[feature_idx]
            # fit each estimator
            estimator.fit(X_subset, y_subset)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_preds = np.empty(shape=(X.shape[0], self.n_estimators))

        for idx, estimator in enumerate(self._estimators):
            feature_idx = self._in_bag_features[idx]
            y_preds[:, idx] = estimator.predict(X[:, feature_idx])

        y_pred = np.empty(shape=X.shape[0])
        for idx, values in enumerate(y_preds):
            y_pred[idx] = self._get_ensemble_value(values)

        return y_pred


    @abc.abstractmethod
    def _get_ensemble_value(self, values: np.ndarray):
        pass
