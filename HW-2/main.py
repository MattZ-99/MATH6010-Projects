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

"""

import math
import os
import numpy as np
from tqdm import tqdm
import argparse


def get_args(*f_args, **f_kwargs):
    """Command-line interfaces, using argpharse module."""

    parser = argparse.ArgumentParser(description='MATH6010-HW-2. monochromatic K4 problem.')
    parser.add_argument('-n', type=int, default=50,
                        help='sum the integers (default: find the max)')

    r_args = parser.parse_args(*f_args, **f_kwargs)

    return r_args


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

        self.edge_color_list = np.zeros(shape=(n, n), dtype=np.int8)  # Initialize adjacent matrix for color edges.

    def _iter_edge(self):
        """Iterator for edge list."""

        for i in range(self.n):
            for j in range(i + 1, self.n):
                yield [i, j]

    @staticmethod
    def _position_slice_generator(n1, n2):
        """Iterator for 4-dim slice."""

        for i in range(4):
            for j in range(i + 1, 4):
                pos = [slice(None)] * 4
                pos[i], pos[j] = n1, n2
                pos = tuple(pos)
                yield pos

    def _iter_k4(self, edge: list):
        for i in range(self.n):
            if i in edge:
                continue
            for j in range(i + 1, self.n):
                if j in edge:
                    continue
                yield [edge[0], edge[1], i, j]

    def _k4_generator(self, edge: list):
        k4_list = list()
        for k4 in self._iter_k4(edge):
            k4_list.append(self.edge_color_list[k4, :][:, k4])
        k4_list = np.stack(k4_list)
        return k4_list

    @staticmethod
    def _calculate_weight_kernel(k4: np.ndarray):
        """ Calculate weights for given k4s. (default white colored)

        :param k4: np.ndarray. array of k4.
        :return: float. sum of weights.
        """

        weight_single = np.sum(np.sum(k4, axis=1), axis=1)
        double_colored = np.min(np.min(k4, axis=1), axis=1)
        weight_single = np.where(double_colored == -1, 0, 2. ** (weight_single - 5))
        weight = np.sum(weight_single)

        return weight

    def _calculate_weight_step(self, edge: list):
        """Calculate relative weight for one step (e.g. for edge i)."""

        k4 = self._k4_generator(edge)

        weight_white = self._calculate_weight_kernel(k4)
        weight_black = self._calculate_weight_kernel(-k4)

        return weight_white > weight_black

    def _update_color_adjacent_matrix(self, edge: list, colored_black: bool):
        """Update color adjacent matrix for one step."""

        if colored_black:
            color = -1
        else:
            color = 1
        self.edge_color_list[edge[0], edge[1]] = color

    def minimum_monochromatic_k4(self):
        """Greedy algorithm for minimum the number of monochromatic K4."""

        for edge in tqdm(self._iter_edge(), total=calculate_comb(n=self.n, k=2)):
            weight_black = self._calculate_weight_step(edge)
            self._update_color_adjacent_matrix(edge, weight_black)

    def _iter_nodes_4(self):
        for i in range(self.n):
            for j in range(i + 1, self.n):
                for k in range(j + 1, self.n):
                    for l in range(k + 1, self.n):
                        yield [i, j, k, l]

    def get_monochromatic_number(self):
        """Final results. Count the number of monochromatic K4."""

        monochromatic_white = 0
        monochromatic_black = 0
        for k4_nodes in self._iter_nodes_4():
            grid = np.ix_(k4_nodes, k4_nodes)
            k4 = self.edge_color_list[grid]
            s = np.sum(k4)
            if s == 6:
                monochromatic_white += 1
            elif s == -6:
                monochromatic_black += 1

        return monochromatic_white + monochromatic_black, monochromatic_white, monochromatic_black


if __name__ == '__main__':
    args = get_args(['-n', '80'])
    n = args.n
    print(calculate_comb(n=n, k=2), calculate_comb(n=n, k=4), int(calculate_comb(n=n, k=4) * (2 ** -5)))
    min_mono_k4 = DerandomizationForMinimumMonochromaticK4(n=n)
    min_mono_k4.minimum_monochromatic_k4()
    result = min_mono_k4.get_monochromatic_number()

    output_dir = "./Outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output = f"n={n}, 总边数={calculate_comb(n=n, k=2)}, 总K4数={calculate_comb(n=n, k=4)}," \
             f" 理论上界={calculate_comb(n=n, k=4) * (2 ** -5)}\n"
    output += f"\t同色K4数={result[0]}, 白色K4数={result[1]}, 黑色K4数={result[2]}\n"

    with open(os.path.join(output_dir, "note.txt"), 'a') as file:
        file.write(output)
