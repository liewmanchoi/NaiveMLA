# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/11/17 10:18
# __file__ = DecisionTree.py

import numpy as np
import abc
import math
from utils.data import is_discrete_target
from sklearn.metrics import accuracy_score


class DecisionTreeNode(object):
    def __init__(self, feature_idx: int = None, cut_off_point: float = None, leaf_output_value=None,
                 left_child: 'DecisionTreeNode' = None, right_child: 'DecisionTreeNode' = None,
                 is_discrete: bool = None):
        self.feature_idx = feature_idx  # 本节点划分所依据的特征索引值
        self.cut_off_point = cut_off_point   # feature_idx处的特征值划分的阈值
        self.leaf_output_value = leaf_output_value  # 叶子节点（如果是叶子节点）的输出值
        self.left_child = left_child  # 左子节点
        self.right_child = right_child  # 右子节点
        self.is_discrete = is_discrete


class BaseDecisionTree(object):
    def __init__(self, max_depth, min_samples_split, min_impurity_split):
        self.root: 'DecisionTreeNode' = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self._n_classes: int = None
        self._n_features: int = None
        self._classes: np.ndarray = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseDecisionTree':
        n_samples, n_features = X.shape
        self._n_features = n_features
        self._classes = np.unique(y)
        self._n_classes = self._classes.size

        self.root = self._generate_tree(X, y, 0)
        return self

    def _generate_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> 'DecisionTreeNode':
        n_samples, n_features = X.shape

        if n_samples < self.min_samples_split or depth >= self.max_depth:
            return DecisionTreeNode(leaf_output_value=self._leaf_value_func(y))

        min_impurity: float = float('inf')
        best_feature_is_discrete: bool = None
        best_feature_idx: int = None
        best_cut_off_feature = None

        # 如果可分样本数或者深度满足要求，则继续下一个步骤
        # 双重循环，选择最优属性和最优切分点
        for feature_idx in range(n_features):
            feature_array = X[:, feature_idx]
            is_discrete = is_discrete_target(feature_array)
            unique_feauture_array = np.unique(feature_array)

            for feature_value in unique_feauture_array:
                if is_discrete:
                    mask = (feature_array == feature_value)
                else:
                    mask = (feature_array <= feature_value)

                y_of_feature = y[mask]
                y_out_of_feature = y[np.logical_not(mask)]

                impurity = self._impurity_func(y_of_feature, y_out_of_feature)

                if impurity < self.min_impurity_split:
                    return DecisionTreeNode(leaf_output_value=self._leaf_value_func(y))

                if impurity < min_impurity:
                    min_impurity = impurity
                    best_feature_is_discrete = is_discrete
                    best_feature_idx = feature_idx
                    best_cut_off_feature = feature_value

        if min_impurity < float("inf"):
            X_left, X_right, y_left, y_right = self._split_dataset(X, y, best_feature_idx,
                                                                   best_cut_off_feature, best_feature_is_discrete)

            left_child = self._generate_tree(X_left, y_left, depth+1)
            right_child = self._generate_tree(X_right, y_right, depth+1)
            root = DecisionTreeNode(feature_idx=best_feature_idx, cut_off_point=best_cut_off_feature,
                                    left_child=left_child, right_child=right_child,
                                    is_discrete=best_feature_is_discrete)
            return root

    @staticmethod
    def _split_dataset(X: np.ndarray, y: np.ndarray, feature_idx: int, cut_off_point, is_discrete: bool):
        if is_discrete:
            mask = (X[:, feature_idx] == cut_off_point)
        else:
            mask = (X[:, feature_idx] <= cut_off_point)
        X_left = X[mask]
        X_right = X[np.logical_not(mask)]
        y_left = y[mask]
        y_right = y[np.logical_not(mask)]

        return X_left, X_right, y_left, y_right

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = np.array([self._predict_value(sample) for sample in X])
        return y_pred

    @abc.abstractmethod
    def _predict_value(self, x: np.ndarray):
        pass

    @abc.abstractmethod
    def _impurity_func(self, y_positive: np.ndarray, y_negative: np.ndarray) -> float:
        pass

    @abc.abstractmethod
    def _leaf_value_func(self, y: np.ndarray):
        pass

    @property
    def n_classes_(self) -> int:
        return self._n_classes

    @property
    def n_features_(self) -> int:
        return self._n_features

    @property
    def classes_(self) -> np.ndarray:
        return self._classes


class DecisionTreeClassifier(BaseDecisionTree):
    def __init__(self, max_depth: int = float("inf"), min_samples_split: int = 2,
                 min_impurity_split: float = 1e-7):
        super().__init__(max_depth, min_samples_split, min_impurity_split)

    # 计算gini_index
    def _impurity_func(self, y_positive: np.ndarray, y_negative: np.ndarray) -> float:
        len1 = y_positive.size
        len2 = y_negative.size
        len_sum = len1 + len2
        return (len1 * self._gini_index(y_positive) + len2 * self._gini_index(y_negative)) / len_sum

    @staticmethod
    def _gini_index(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0

        y_unique = np.unique(y)
        sum_of_power_of_p = 0.0
        for label in y_unique:
            p = (label == y).sum() / y.size
            sum_of_power_of_p += math.pow(p, 2)

        gini_index = 1 - sum_of_power_of_p
        return gini_index

    def _leaf_value_func(self, y: np.ndarray) -> int:
        return int(np.argmax(np.bincount(y)))

    def _predict_value(self, x: np.ndarray) -> int:
        node: 'DecisionTreeNode' = self.root

        while node.leaf_output_value is None:
            feature_value = x[node.feature_idx]
            if node.is_discrete:
                if feature_value == node.cut_off_point:
                    node = node.left_child
                else:
                    node = node.right_child
            else:
                if feature_value <= node.cut_off_point:
                    node = node.left_child
                else:
                    node = node.right_child

        return node.leaf_output_value

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))
