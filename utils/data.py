# -*- coding: utf-8 -*-
# __author__ = wangsheng
# __copyright__ = "Copyright 2018, Trump Organization"
# __email__ = "liewmanchoi@gmail.com"
# __status__ = "experiment"
# __time__ = 2018/11/19 15:21
# __file__ = data.py

import numpy as np


def is_discrete_target(array: np.ndarray) -> bool:
    if array.dtype.kind == 'f' and np.any(array != array.astype(int)):
        return False

    return True
