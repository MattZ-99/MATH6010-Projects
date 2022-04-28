# -*- coding: utf-8 -*-
# @Time : 2022/4/28 11:09
# @Author : Mengtian Zhang
# @Version : v-dev-0.0
# @Function 

"""A hill-climbing algorithm for Steiner Triple Systems. STS(v).

Description.----------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------

Example:

"""

import argparse
import random
import time
from typing import Set, Tuple


class STSSystem:
    def __init__(self, v: int = 3):
        """Initialize a STS system.

        :param v: positive int & v = 1, 3 (mod 6).
        """

        super(STSSystem, self).__init__()
        assert v >= 3 and (v % 6 == 1 or v % 6 == 3)

        self.set_v = set(i for i in range(v))
        self.adjacent_dict = {k: set() for k in self.set_v}
        self.block_num = v * (v - 1) / 6
        self.current_block_num = 0
        self.v = v
        self.r = (v - 1) / 2
        self.current_blocks = set()

    def hill_climbing_alg_for_STSv(self):
        """Hill-climbing algorithm for creating an STS blocks set."""

        while self.current_block_num < self.block_num:
            x = self._generate_random_live_vertex()
            y, z = self._select_pair_vertex_from_x(x)

            if self._check_pair_live(y, z):
                self._add_block(x, y, z)
                self.current_block_num += 1
            else:
                self._delete_block_with_pair(y, z)
                self._add_block(x, y, z)

    def _add_block(self, x, y, z):
        """Add a block in the current blocks set and update the adjacent dict."""

        self.adjacent_dict[x].update([y, z])
        self.adjacent_dict[y].update([x, z])
        self.adjacent_dict[z].update([x, y])

        self.current_blocks.add(tuple(sorted([x, y, z])))

        return True

    def _delete_block(self, x, y, z):
        """Delete a block in the current blocks set and update the adjacent dict."""

        self.adjacent_dict[x] -= {y, z}
        self.adjacent_dict[y] -= {x, z}
        self.adjacent_dict[z] -= {x, y}

        self.current_blocks.remove(tuple(sorted([x, y, z])))

        return True

    def _get_third_vertex_from_pair(self, y, z):
        """Get the third block vertex for the given pair."""

        vl = [y, z]
        blk = None
        for b in self.current_blocks:
            if all([v in b for v in vl]):
                blk = b
                break

        w = (set(blk) - set(vl)).pop()
        return w

    def _delete_block_with_pair(self, y, z):
        """Delete the block containing the given pair."""

        w = self._get_third_vertex_from_pair(y, z)
        return self._delete_block(w, y, z)

    def _generate_random_live_vertex(self):
        """Generate a live vertex randomly from the vertex set."""

        while True:
            selected = random.sample(self.set_v, 1)[0]
            selected_blocks = len(self.adjacent_dict[selected]) / 2
            if selected_blocks < self.r:
                return selected

    def _select_pair_vertex_from_x(self, x):
        """Select a pair of vertexes from x's adjacency randomly.

        :param x: a given vertex.
        :return: a pair of vertexes from x's adjacency.
        """

        live_vertex = self.set_v - self.adjacent_dict[x]
        live_vertex.remove(x)

        selected_pair = random.sample(live_vertex, 2)
        return selected_pair[0], selected_pair[1]

    def _check_pair_live(self, y, z) -> bool:
        """ Check whether the given pair of vertexes is live.

        :param y: the first vertex.
        :param z: the second vertex.
        :return: bool. Whether the pair is live.
        """
        return not ((z in self.adjacent_dict[y]) and (y in self.adjacent_dict[z]))

    def get_current_PSTS(self) -> Set[Tuple[int]]:
        """ Get the current PSTS block set.

        :return: Current block set.
        """

        return self.current_blocks


def save_arg_parameters(_args, path):
    """Save the program parameters into the file."""

    lines = "Parameters:\n"
    parameters = vars(_args)
    for key in parameters:
        line = f"{key} -> {parameters[key]}"
        lines += line + '\n'

    split_str = '\n' * 5
    split_star = '*' * 100 + '\n'
    split_str = split_str + split_star * 3
    with open(path, 'a') as file:
        file.write(split_str)
        file.write(lines)


def save_output(_output: str, path):
    """Save the output into the file."""

    split_str = '=' * 50 + '\n'
    with open(path, 'a') as file:
        file.write(split_str)
        file.write(_output)


def set_seed(seed):
    """Set seed for random, numpy."""

    import os
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    # parameters
    save_file_path = "./Outputs/result-hill-climbing-for-STSv.txt"
    parameter_dict = {
        'v': 63,
        'seed': 220505
    }

    args = argparse.Namespace(**parameter_dict)
    print(args)

    # Save parameters.
    save_arg_parameters(args, save_file_path)

    # Set seed.
    set_seed(args.seed)

    # Running.
    t1 = time.time()
    sts_v = STSSystem(v=args.v)
    sts_v.hill_climbing_alg_for_STSv()
    sts_blocks = sts_v.get_current_PSTS()
    t2 = time.time()

    # Output.
    output = ""
    output += "* " + "Algorithm running time: {:.3f}s.".format(t2 - t1) + '\n'
    output += "* " + "Number of STSv blocks: {}.\t Theory STSv blocks |B| = v(v-1)/6 = {}.".format(
        len(sts_blocks), int(args.v * (args.v - 1) / 6)) + '\n'

    occ_times = [len(v) / 2 for v in sts_v.adjacent_dict.values()]
    output += "* " + "Every point occur times: {}. \t Theory occur times r = (v-1)/2 = {}.".format(
        int((min(occ_times) + max(occ_times)) / 2), int((args.v - 1) / 2)) + '\n'
    output += "* " + f"Blocks set: {sts_blocks}" + '\n'

    print(output)
    save_output(output, save_file_path)
