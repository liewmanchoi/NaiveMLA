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
from supervised_learning.gradientboosting import MeanEstimator
from supervised_learning.gradientboosting import LogOddsEstimator
from scipy.special import expit
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score


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
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        pass

    @property
    def gradient(self) -> np.ndarray:
        # 计算一阶偏导数
        return self._gradient

    @property
    def hess(self) -> np.ndarray:
        # 计算二阶偏导数
        return self._hess


class LeastSquareError(LossFunction):
    def init_estimator(self):
        return MeanEstimator()

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self._gradient = y_pred - y_true
        self._hess = np.ones(shape=len(y_true))


class BinomialDeviance(LossFunction):
    def init_estimator(self):
        return LogOddsEstimator()

    # L(y_true, y_pred) = y_true * log(1 + exp(-1 * y_pred)) + (1 - y_true) * log(1 + exp(y_pred))
    # y_true - {0, 1}
    # P(y=1 | x) = 1 / (1 + exp(-y_pred)) = expit(y_pred)
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        # dL / df(x) = 1 / (1 + exp(-f(x)) - y_true = expit(y_pred) - y_true
        # d^2 L / d^2 f(x) = 1 / ((1 + exp(f(x))(1 + exp(-f(x))) = expit(-y_pred) * expit(y_pred)
        self._gradient = expit(y_pred) - y_true
        self._hess = expit(y_pred) * expit(-y_pred)

    @staticmethod
    def output_to_proba(y_pred: np.ndarray) -> np.ndarray:
        proba = expit(y_pred)
        return proba


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

            if node.depth == self._max_depth:
                node.leaf_output_value = self._leaf_output(index, gradient, hess)
                continue

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
                    continue

                for feature in unique_features:
                    if is_discrete:
                        mask = (features == feature)
                    else:
                        mask = (features <= feature)

                    left_index = index[mask]
                    right_index = index[~mask]

                    if len(left_index) == 0 or len(right_index) == 0:
                        continue

                    gain = self._gain(left_index, right_index, gradient, hess)
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
        g: np.ndarray = gradient[index]
        h: np.ndarray = hess[index]
        weight: float = - g.sum() / (h.sum() + self._reg_lambda)

        return weight

    def _gain(self, left_index: np.ndarray, right_index: np.ndarray,
              gradient: np.ndarray, hess: np.ndarray) -> float:
        left_gradient_sum = gradient[left_index].sum()
        left_hess_sum = hess[left_index].sum()
        right_gradient_sum = gradient[right_index].sum()
        right_hess_sum = hess[right_index].sum()
        gradient_sum = left_gradient_sum + right_gradient_sum
        hess_sum = left_hess_sum + right_hess_sum

        post_split_loss = \
            math.pow(left_gradient_sum, 2) / (left_hess_sum + self._reg_lambda) + \
            math.pow(right_gradient_sum, 2) / (right_hess_sum + self._reg_lambda)
        pre_split_loss = math.pow(gradient_sum, 2) / (hess_sum + self._reg_lambda)
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


class BaseXGBoost(object):
    def __init__(self, loss: 'LossFunction', max_depth: int, learning_rate: float, n_estimators: int, gamma: float,
                 reg_lambda: float):
        self._loss: 'LossFunction' = loss
        self._max_depth: int = max_depth
        self._learning_rate: float = learning_rate
        self._n_estimators: int = n_estimators
        self._estimators: List['XGBDecisionTreeRegressor'] = list()
        self._gamma: float = gamma
        self._reg_lambda: float = reg_lambda
        self._init = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseXGBoost':
        # init model & self._estimators
        self._init_state()
        self._init.fit(X, y)
        # 赋予初值
        f = self._init_decision_function(X)
        # 每个回归器使用牛顿法拟合
        for estimator in self._estimators:
            # 计算损失函数的一阶偏导（梯度）和二阶偏导
            self._loss.compute(y_true=y, y_pred=f)
            estimator.fit(X, gradient=self._loss.gradient, hess=self._loss.hess)
            # 注意shrinkage机制
            f += self._learning_rate * estimator.predict(X)

        return self

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        # 回归器可以直接使用本函数，分类器还需要转换为概率和类别
        f = self._init_decision_function(X)

        for estimator in self._estimators:
            f += self._learning_rate * estimator.predict(X)

        return f

    def _init_state(self) -> None:
        self._init = self._loss.init_estimator()
        for _ in range(self._n_estimators):
            self._estimators.append(XGBDecisionTreeRegressor(max_depth=self._max_depth, gamma=self._gamma,
                                                             reg_lambda=self._reg_lambda))

    def _init_decision_function(self, X: np.ndarray) -> np.ndarray:
        score = self._init.predict(X).astype(np.float64)
        return score

    @abc.abstractmethod
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        pass


class XGBRegressor(BaseXGBoost):
    def __init__(self, max_depth: int = 3, learning_rate: float = 0.1, n_estimators: int = 100, gamma: float = 0,
                 reg_lambda: float = 1):
        super().__init__(loss=LeastSquareError(), max_depth=max_depth, learning_rate=learning_rate,
                         n_estimators=n_estimators, gamma=gamma, reg_lambda=reg_lambda)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return super().predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return r2_score(y, self.predict(X))


class XGBClassifier(BaseXGBoost):
    def __init__(self, max_depth: int = 3, learning_rate: float = 0.1, n_estimators: int = 100, gamma: float = 0,
                 reg_lambda: float = 1):
        super().__init__(loss=BinomialDeviance(), max_depth=max_depth, learning_rate=learning_rate,
                         n_estimators=n_estimators, gamma=gamma, reg_lambda=reg_lambda)

    def predict(self, X: np.ndarray) -> np.ndarray:
        f = super().predict(X)
        proba = np.empty(shape=(X.shape[0], 2))
        proba[:, 1] = BinomialDeviance.output_to_proba(f)
        proba[:, 0] = 1 - proba[:, 1]
        y_pred = np.argmax(proba, axis=1)

        return y_pred

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))
