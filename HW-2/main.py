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
    min_v = min(k, n - k)
    max_v = max(k, n - k)
    mult = 1
    for i in range(max_v + 1, n + 1):
        mult *= i
    mult /= math.factorial(min_v)
    return int(mult)


class DerandomizationForMinimumMonochromaticK4(object):
    """Derandomization method (greedy) for monochromatic K4.

    Upper bound: a two-coloring of Kn with at most (n 4)2^{-5} monochromatic K4.
    """

    def __init__(self, n: int = 10):
        """Class initialization.

        Initialize weight array, colored edge num for each K4, edge color list for adjacent matrix.

        :param n: int. node numbers in Kn.
        """

        super(DerandomizationForMinimumMonochromaticK4, self).__init__()
        self.__heterochromatic_value = 100
        self.n = n

        self.colored_edge_num = np.zeros(shape=(n, n, n, n), dtype=np.int8)  # Init colored edger number.

        self.true_k4_pos = np.zeros(shape=(n, n, n, n), dtype=np.bool_)
        coordinate = np.linspace(0, n, n, endpoint=False, dtype=np.int8)
        n1, n2, n3, n4 = np.meshgrid(coordinate, coordinate, coordinate, coordinate, indexing='ij', copy=False)
        self.true_k4_pos[(n1 < n2) & (n2 < n3) & (n3 < n4)] |= True

        self.edge_color_list = np.zeros(shape=(n, n), dtype=np.int8)  # Initialize adjacent matrix for color edges.

    def _iter_edge(self):
        """Iterator for edge list."""

        for i in range(self.n):
            for j in range(i + 1, self.n):
                yield [i, j]

    def _position_slice_generator(self, n1, n2):
        """Iterator for 4-dim slice."""

        for i in range(4):
            for j in range(i + 1, 4):
                pos = [slice(None)] * 4
                pos[i], pos[j] = n1, n2
                pos = tuple(pos)
                yield pos

    def _calculate_weight_step(self, edge: list):
        """Calculate relative weight for one step (e.g. for edge i)."""

        n1, n2 = edge[0], edge[1]
        color_mat = list()
        pos_true_mat = list()

        for pos in self._position_slice_generator(n1, n2):
            colored_edge_num = self.colored_edge_num[pos]
            color_mat.append(colored_edge_num)
            pos_true = self.true_k4_pos[pos]
            pos_true_mat.append(pos_true)

        colored_edge_num = np.concatenate(color_mat, axis=0)
        pos_true_mat = np.concatenate(pos_true_mat, axis=0)

        weight_white = np.sum(2. **
                              (colored_edge_num[(colored_edge_num >= 0) & pos_true_mat
                                                & (colored_edge_num != self.__heterochromatic_value)] - 5))
        weight_black = np.sum(2. **
                              (-colored_edge_num[(colored_edge_num <= 0) & pos_true_mat
                                                 & (colored_edge_num != self.__heterochromatic_value)] - 5))

        return weight_white > weight_black

    def _update_step(self, edge: list, colored_black: bool):
        """Update weight array and colored edge num for one step."""

        n1, n2 = edge[0], edge[1]

        for pos in self._position_slice_generator(n1, n2):

            if colored_black:
                self.colored_edge_num[pos] = np.where((self.colored_edge_num[pos] <= 0) & (self.true_k4_pos[pos])
                                                      & (self.colored_edge_num[pos] != self.__heterochromatic_value),
                                                      self.colored_edge_num[pos] - 1, self.__heterochromatic_value
                                                      )
            else:
                self.colored_edge_num[pos] = np.where((self.colored_edge_num[pos] >= 0) & (self.true_k4_pos[pos])
                                                      & (self.colored_edge_num[pos] != self.__heterochromatic_value),
                                                      self.colored_edge_num[pos] + 1, self.__heterochromatic_value
                                                      )

    def _update_color_adjacent_matrix(self, edge: list, colored_black: bool):
        """Update color adjacent matrix for one step."""

        if colored_black:
            color = -1
        else:
            color = 1
        self.edge_color_list[edge[0], edge[1]] = color
        self.edge_color_list[edge[1], edge[0]] = color

    def minimum_monochromatic_k4(self):
        """Greedy algorithm for minimum the number of monochromatic K4."""

        for edge in tqdm(self._iter_edge(), total=calculate_comb(n=self.n, k=2)):
            weight_black = self._calculate_weight_step(edge)
            self._update_step(edge, weight_black)
            self._update_color_adjacent_matrix(edge, weight_black)

    def get_monochromatic_number(self):
        """Final results. Count the number of monochromatic K4."""

        monochromatic_white = np.sum((self.colored_edge_num == 6))
        monochromatic_black = np.sum((self.colored_edge_num == -6))
        return monochromatic_white + monochromatic_black, monochromatic_white, monochromatic_black


if __name__ == '__main__':
    n = 200
    print(calculate_comb(n=n, k=2), calculate_comb(n=n, k=4), calculate_comb(n=n, k=4) * (2 ** -5))
    min_mono_k4 = DerandomizationForMinimumMonochromaticK4(n=n)
    min_mono_k4.minimum_monochromatic_k4()
    print(min_mono_k4.get_monochromatic_number())
