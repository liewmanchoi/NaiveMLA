# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/11/17 10:18
# __file__ = DecisionTree.py

from typing import Callable


class DecisionTreeNode(object):
    def __init__(self, feature_idx: int = None, threshold: float = None, output_value: float = None,
                 left_child: 'DecisionTreeNode' = None, right_child: 'DecisionTreeNode' = None):
        self.feature_idx = feature_idx  # 本节点划分所依据的特征索引值
        self.threshold = threshold   # feature_idx处的特征值划分的阈值
        self.output_value = output_value  # 叶子节点（如果是叶子节点）的输出值
        self.left_child = left_child  # 左子节点
        self.right_child = right_child  # 右子节点


class BaseDecisionTree(object):
    def __init__(self, max_depth: int = float("inf"), min_samples_split: int = 2,
                 min_impurity_split: float = 1e-7):
        self.root: 'DecisionTreeNode' = None
        self.max_depty = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self._impurity_func: Callable = None
        self._left_value_func: Callable = None

