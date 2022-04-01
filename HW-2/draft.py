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

a = np.ones(5)
b = a[1]
b += 1
print(a)
a[[1, 3]][1] += 1
print(a)
