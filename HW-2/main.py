# -*- coding: utf-8 -*-
# @Time : 2022/4/1 11:40
# @Author : Mengtian Zhang
# @Version : v-dev-0.0
# @Function

""" SJTU-MATH6010-图与网络 Homework-2

Derandomization.
Theorem: There is a two-coloring of Kn with at most (n 4)2^{-5} monochromatic K4.
问题描述：用条件概率方法实现边染色，使得最终得到的同色K4的数目小于等于期望值。

Example:

Todo:
    * For module TODOs

"""

import math
from functools import lru_cache
import numpy as np
import time
from tqdm import tqdm


def calculate_comb(n: int, k: int) -> int:
    """Given n & k, calculate combination number."""

    if (k > n) or (n < 0) or (k < 0):
        raise ValueError
    return int(math.factorial(n) / math.factorial(k) / math.factorial(n - k))


# def minimum_monochromatic_k4(n: int = 10) -> None:
#
#     k4_num = calculate_comb(n, 4)
#     weight_array = (2 ** (-5)) * np.ones(k4_num)  # Initialize weight for every K4
#     colored_edge_num = np.zeros(k4_num)
#
#     for e in range(k4_num):


class DerandomizationForMinimumMonochromaticK4(object):
    """Derandomization method (greedy) for monochromatic K4.

    Upper bound: a two-coloring of Kn with at most (n 4)2^{-5} monochromatic K4.
    """

    def __init__(self, n: int = 10):
        super(DerandomizationForMinimumMonochromaticK4, self).__init__()
        self.n = n
        k4_num = calculate_comb(n, 4)
        self.weight_array = (2 ** (-5)) * np.ones(k4_num)  # Initialize weight for every K4.
        self.colored_edge_num = np.zeros(k4_num).astype(int)  # Init colored edger number.
        self.edge_color_list = None  # Color list for edges.

    def _iter_edge(self):
        for i in range(self.n):
            for j in range(i + 1, self.n):
                yield [i, j]

    def _iter_k4(self, edge: list):
        for i in range(self.n):
            if i in edge:
                continue
            for j in range(i + 1, self.n):
                if j in edge:
                    continue
                yield [i, j, edge[0], edge[1]]

    def _calculate_k4_array_pos(self, nodes: list):
        k4 = sorted(nodes)
        k4.insert(0, -1)
        number = 0
        for i in range(len(k4) - 1):
            for j in range(k4[i] + 1, k4[i + 1]):
                number += calculate_comb(self.n - 1 - j, 3 - i)
        return number

    def minimum_monochromatic_k4(self):
        edge_color_list = list()
        for edge in self._iter_edge():
            pos = []
            for k4 in self._iter_k4(edge):
                pos.append(self._calculate_k4_array_pos(k4))

            weight_past = self.weight_array[pos]
            colored_edge_num = self.colored_edge_num[pos]

            weight_white = np.sum(weight_past[colored_edge_num == 0]) + \
                           np.sum(weight_past[(colored_edge_num > 0) & (weight_past > 0)]) * 2
            weight_black = np.sum(weight_past[colored_edge_num == 0]) + \
                           np.sum(weight_past[(colored_edge_num > 0) & (weight_past < 0)]) * (-2)
            
            if weight_white > weight_black:
                weight_past[colored_edge_num == 0] *= -1
                weight_past[(colored_edge_num > 0) & (weight_past > 0)] = 0
                weight_past[(colored_edge_num > 0) & (weight_past < 0)] *= 2
                edge_color_list.append(0)
            else:
                weight_past[(colored_edge_num > 0) & (weight_past > 0)] *= 2
                weight_past[(colored_edge_num > 0) & (weight_past < 0)] = 0
                edge_color_list.append(1)

            self.weight_array[pos] = weight_past
            self.colored_edge_num[pos] += 1

        self.edge_color_list = edge_color_list
        
        result = np.sum(np.where(self.weight_array == 0, 0, 1))
        return result


if __name__ == '__main__':
    min_mono_k4 = DerandomizationForMinimumMonochromaticK4(n=50)
    print(calculate_comb(n=50, k=2))
    result = min_mono_k4.minimum_monochromatic_k4()
    print(result)
