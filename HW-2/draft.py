# -*- coding: utf-8 -*-
# @Time : 2022/4/1 12:16
# @Author : Mengtian Zhang
# @Version : v-dev-0.0
# @Function 

"""Summary.

Description.----------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------

Example:

Todo:
    * For module TODOs

"""

import numpy as np

a = np.zeros(10, dtype=np.bool_)
print(~a)
a = a | True
print(a == 1)
