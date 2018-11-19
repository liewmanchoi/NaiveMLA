# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/11/17 10:18
# __file__ = DecisionTree.py

import numpy as np
import abc


class DecisionTreeNode(object):
    def __init__(self, feature_idx: int = None, cut_off_point: float = None, leaf_output_value=None,
                 left_child: 'DecisionTreeNode' = None, right_child: 'DecisionTreeNode' = None):
        self.feature_idx = feature_idx  # 本节点划分所依据的特征索引值
        self.cut_off_point = cut_off_point   # feature_idx处的特征值划分的阈值
        self.leaf_output_value = leaf_output_value  # 叶子节点（如果是叶子节点）的输出值
        self.left_child = left_child  # 左子节点
        self.right_child = right_child  # 右子节点


class BaseDecisionTree(object):
    def __init__(self, max_depth: int = float("inf"), min_samples_split: int = 2,
                 min_impurity_split: float = 1e-7):
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

        min_impurity: float = float('inf')
        best_feature_idx: int = None
        best_cut_off_feature = None

        # 如果可分样本数或者深度满足要求，则继续下一个步骤
        if n_samples >= self.min_samples_split and depth < self.max_depth:
            # 双重循环，选择最优属性和最优切分点
            for feature_idx in range(n_features):
                feature_array = X[:, feature_idx]
                unique_feauture_array = np.unique(feature_array)

                # case1: 所有样本在所有属性上取值相同，无法划分
                if unique_feauture_array.size == 1:
                    return DecisionTreeNode(leaf_output_value=self._leaf_value_func(y))

                for feature in unique_feauture_array:
                    mask: np.ndarray = (feature_array == feature)
                    y_of_feature = y[mask]
                    y_out_of_feature = y[np.logical_not(mask)]

                    # case2: 当前结点包含的样本全部属于同一类别
                    # case3: 当前结点包含的样本集合为空
                    if len(y_of_feature) == 0 or len(y_out_of_feature) == 0:
                        return DecisionTreeNode(leaf_output_value=self._leaf_value_func(y))

                    impurity = self._impurity_func(y_of_feature, y_out_of_feature)

                    if impurity < min_impurity:
                        min_impurity = impurity
                        best_feature_idx = feature_idx
                        best_cut_off_feature = feature

            # 如果最小的不纯度高于预定阈值，则可以进行特征切分
            if self.min_impurity_split < min_impurity < float("inf"):
                X_left, X_right, y_left, y_right = self._split_dataset(X, y, best_feature_idx, best_cut_off_feature)
                left_child = self._generate_tree(X_left, y_left, depth+1)
                right_child = self._generate_tree(X_right, y_right, depth+1)
                root = DecisionTreeNode(feature_idx=best_feature_idx, cut_off_point=best_cut_off_feature,
                                        left_child=left_child, right_child=right_child)
                return root

        return DecisionTreeNode(leaf_output_value=self._leaf_value_func(y))

    @staticmethod
    def _split_dataset(X: np.ndarray, y: np.ndarray, feature_idx: int, cut_off_point):
        n_features = X.shape[1]
        exclude_mask = np.delete(np.arange(n_features), feature_idx)
        X_exclude = X.take(exclude_mask, axis=1)
        mask = X[:, feature_idx] == cut_off_point
        X_left = X_exclude[mask]
        X_right = X_exclude[np.logical_not(mask)]
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
