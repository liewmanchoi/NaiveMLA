# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/12/13 15:00
# __file__ = xgboost.py

import numpy as np
import abc
import math
from typing import List
from utils.data import is_discrete_target
from supervised_learning.GradientBoosting import LogOddsEstimator
from supervised_learning.GradientBoosting import LeastSquareError
from supervised_learning.GradientBoosting import BinomialDeviance


class DecisionTreeNode(object):
    def __init__(self, depth: int, feature_idx: int = None, cut_off_point=None, is_discrete: bool = None,
                 leaf_output_value=None, left_child: 'DecisionTreeNode' = None, right_child: 'DecisionTreeNode' = None):
        self.depth: int = depth
        self.feature_idx: int = feature_idx  # 本节点划分所依据的特征索引值
        self.cut_off_point = cut_off_point  # feature_idx处的特征值划分的阈值
        self.is_discrete: bool = is_discrete
        self.leaf_output_value: float = leaf_output_value  # 叶子节点（如果是叶子节点）的输出值
        self.left_child: 'DecisionTreeNode' = left_child  # 左子节点
        self.right_child: 'DecisionTreeNode' = right_child  # 右子节点


class LossFunction(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._gradient: np.ndarray = None
        self._hess: np.ndarray = None

    @abc.abstractmethod
    def init_estimator(self):
        # init estimator for loss function
        pass

    @abc.abstractmethod
    def __call__(self, y_true, y_pred):
        # compute the loss
        pass

    @property
    def gradient(self) -> np.ndarray:
        # 计算一阶偏导数
        return self._gradient

    @property
    def hess(self) -> np.ndarray:
        # 计算二阶偏导数
        return self._hess


class XGBDecisionTreeRegressor(object):
    # reference: http://arxiv.org/abs/1603.02754
    def __init__(self, max_depth: int, gamma: float, reg_lambda: float):
        self._root: 'DecisionTreeNode' = None
        self._max_depth: int = max_depth
        # Minimum loss reduction required to make a further partition on a leaf node of the tree
        self._gamma: float = gamma
        self._reg_lambda: float = reg_lambda
        self._n_features: int = None
        self._n_samples: int = None

    def fit(self, X: np.ndarray, gradient: np.ndarray, hess: np.ndarray) -> 'XGBDecisionTreeRegressor':
        self._n_samples, self._n_features = X.shape

        self._root = self._build_tree(X, gradient, hess)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        y_pred = np.array([self._predict_value(sample) for sample in X])

        return y_pred

    def _build_tree(self, X: np.ndarray, gradient: np.ndarray, hess: np.ndarray) -> 'DecisionTreeNode':
        depth: int = 0
        root: 'DecisionTreeNode' = DecisionTreeNode(depth=depth)

        nodes_stack: List['DecisionTreeNode'] = [root]
        index_stack: List[np.ndarray] = [np.arange(self._n_samples)]

        while len(nodes_stack) > 0:
            node = nodes_stack.pop()
            index = index_stack.pop()

            if node.depth >= self._max_depth:
                node.leaf_output_value = self._leaf_output(index, gradient, hess)

            max_gain: float = -float('inf')
            best_feature_idx: int = None
            best_feature_is_discrete: bool = None
            best_cut_off_point = None
            post_left_index: np.ndarray = None
            post_right_index: np.ndarray = None

            for feature_idx in np.arange(self._n_features):
                features = X[index, feature_idx]
                unique_features = np.unique(features)
                is_discrete = is_discrete_target(unique_features)

                if len(unique_features) == 1:
                    node.leaf_output_value = self._leaf_output(index, gradient, hess)

                for feature in unique_features:
                    if is_discrete:
                        mask = (features == feature)
                    else:
                        mask = (features <= feature)

                    left_index = index[mask]
                    right_index = index[~mask]

                    if len(left_index) == 1 or len(right_index) == 1:
                        continue

                    gain = self._gain(index, left_index, right_index, gradient, hess)
                    if gain > max_gain:
                        max_gain = gain
                        best_feature_idx = feature_idx
                        best_feature_is_discrete = is_discrete
                        best_cut_off_point = feature
                        post_left_index = left_index
                        post_right_index = right_index

            # 如果最大增益大于阈值self._gamma，则分裂节点
            if max_gain > self._gamma:
                node.feature_idx = best_feature_idx
                node.cut_off_point = best_cut_off_point
                node.is_discrete = best_feature_is_discrete
                left_child = DecisionTreeNode(node.depth + 1)
                right_child = DecisionTreeNode(node.depth + 1)
                node.left_child = left_child
                node.right_child = right_child

                nodes_stack.append(right_child)
                index_stack.append(post_right_index)
                nodes_stack.append(left_child)
                index_stack.append(post_left_index)

            else:
                node.leaf_output_value = self._leaf_output(index, gradient, hess)

        return root

    def _leaf_output(self, index: np.ndarray, gradient: np.ndarray, hess: np.ndarray) -> float:
        gradient: np.ndarray = gradient[index]
        hess: np.ndarray = hess[index]
        weight: float = - gradient.sum() / (hess.sum() + self._reg_lambda)

        return weight

    def _gain(self, index: np.ndarray, left_index: np.ndarray, right_index: np.ndarray,
              gradient: np.ndarray, hess: np.ndarray) -> float:
        left_gradient = gradient[left_index]
        left_hess = hess[left_index]
        right_gradient = gradient[right_index]
        right_hess = hess[right_index]
        gradient = gradient[index]
        hess = hess[index]

        post_split_loss = math.sqrt(left_gradient.sum()) / (left_hess.sum() + self._reg_lambda) + \
                          math.sqrt(right_gradient.sum()) / (right_hess.sum() + self._reg_lambda)
        pre_split_loss = math.sqrt(gradient.sum()) / (hess.sum() + self._reg_lambda)
        gain = (post_split_loss - pre_split_loss) / 2

        return gain

    def _predict_value(self, x: np.ndarray):
        node = self._root

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
