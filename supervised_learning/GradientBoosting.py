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

