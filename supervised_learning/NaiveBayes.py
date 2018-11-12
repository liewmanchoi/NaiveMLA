# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/11/12 9:46
# __file__ = NaiveBayes.py

import numpy as np
from sklearn.metrics import accuracy_score


class GaussianNB(object):
    def __init__(self, var_smoothing=1e-09):
        self._priors: np.ndarray = None
        self._means: np.ndarray = None
        self._vars: np.ndarray = None
        self._classes: np.ndarray = None
        self.n_classes_: int = 0
        self.n_features_: int = 0
        self._var_smoothing = var_smoothing

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNB':
        n_samples, self.n_features_ = X.shape
        self._classes = np.unique(y)
        self.n_classes_ = self._classes.size

        self._means = np.empty(shape=(self.n_classes_, self.n_features_), dtype=np.float64)
        self._vars = np.empty(shape=(self.n_classes_, self.n_features_), dtype=np.float64)
        self._priors = np.empty(shape=self.n_classes_, dtype=np.float64)

        for i, c in enumerate(self._classes):
            X_where_c = X[y == c]
            self._means[i] = X_where_c.mean(axis=0)
            self._vars[i] = X_where_c.var(axis=0)
            self._priors[i] = np.mean(y == c)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_predict = np.apply_along_axis(func1d=self._classify, axis=1, arr=X)
        return y_predict

    def _calculate_log_likehood_matrix(self, x: np.ndarray) -> np.ndarray:
        numerator = -1 * np.power(x - self._means, 2) / (2 * self._vars + self._var_smoothing)
        denominator = np.log(np.sqrt(2 * np.pi * self._vars + self._var_smoothing))

        return numerator - denominator

    def _classify(self, x: np.ndarray) -> int:
        posteriors = list()

        log_likehood_matrix = self._calculate_log_likehood_matrix(x)
        for i, log_likehood_vector in zip(range(self.n_classes_), log_likehood_matrix):
            posterior = log_likehood_vector.sum() + np.log(self._priors[i])
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))
