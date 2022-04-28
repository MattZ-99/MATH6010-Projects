# -*- coding: utf-8 -*-
# @Time : 2022/4/28 11:10
# @Author : Mengtian Zhang
# @Version : v-dev-0.0
# @Function 

"""Uniform graph partition.

Description.----------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------

Example:

"""

from functools import wraps
import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import os
import argparse


def set_seed(seed):
    """Set seed for random, numpy."""

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def comb(n, r):
    """Calculate the combination number of C_n^r"""

    f = math.factorial
    return f(n) / f(r) / f(n - r)


def timer(func):
    """Decorator for timing."""

    @wraps(func)
    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        output_star = "*" * 60
        output = '\n*  {} cost time {:.3f} s\n'.format(func.__name__, time_spend)
        output = output_star + output + output_star

        print("\033[36m" + output + "\033[0m")

        return result, time_spend

    return func_wrapper


class UniformGraphPartition(object):
    """Solution class for uniform graph partition problem."""

    def __init__(self, node_num_half: int = None):
        """Initialize the UniformGraphPartition class.

        :param node_num_half: node numbers for one part of graph.
        """

        super(UniformGraphPartition, self).__init__()
        self.graph = None
        self.generate_complete_graph(node_num_half * 2)
        self.set_random_edge_weight()
        self._random_generate_init_partition()

    def generate_complete_graph(self, num: int = None):
        """Generate the complete graph.

        :param num: the number of graph nodes.
        """

        if num is None:
            self.graph = nx.Graph()
        self.graph = nx.complete_graph(num)

    def set_random_edge_weight(self):
        """Randomly set the weights of edges."""

        num_of_node = self.graph.number_of_nodes()
        # weights = np.around(np.random.randn(num_of_node, num_of_node), 2)
        weights = np.random.randint(0, 100, size=(num_of_node, num_of_node))
        # weights = np.ones(shape=(num_of_node, num_of_node))
        for e in self.graph.edges:
            self.graph[e[0]][e[1]]['weight'] = weights[e[0]][e[1]]

    def _random_generate_init_partition(self):
        """Randomly generate an initial partition of graph nodes."""

        gra = self.graph
        part_1, part_2 = self._random_split_of_nodes(gra.nodes)
        self.init_part = (part_1, part_2)

    def get_init_partition(self):
        """Get the initial nodes partition with copy.

        :return: the copy of part_1, part_2.
        """

        part_1, part_2 = self.init_part
        return part_1.copy(), part_2.copy()

    def plot_graph(self, pos_name: str = None, *args_, **kwargs_):
        """Plot the graph."""

        gra = self.graph

        # plot graph only in small size.
        if gra.number_of_nodes() > 10:
            return

        if pos_name is None:
            pos = nx.spring_layout(gra)
        else:
            if pos_name == 'bipartite':
                pos = nx.bipartite_layout(gra, kwargs_['first_part'])
            else:
                raise NotImplementedError

        labels = nx.get_edge_attributes(gra, 'weight')

        fig, ax = plt.subplots()
        nx.draw_networkx(gra, pos, ax=ax)
        nx.draw_networkx_edge_labels(gra, pos, edge_labels=labels, label_pos=0.4, ax=ax)
        ax.axis('off')

        if pos_name == 'bipartite':
            ax.text(0.1, -0.1, f"* First part: {kwargs_['first_part']}", transform=ax.transAxes, fontsize="large")
            ax.text(0.1, -0.2, f"* Second part: {kwargs_['second_part']}", transform=ax.transAxes, fontsize="large")

        plt.tight_layout()
        plt.show()
        plt.close('all')

    def output_graph(self):
        """Output the basic parameters of the graph."""

        gra = self.graph

        print(f"Nodes: {gra.nodes}.")
        print(f"Edges: {gra.edges(data=True)}.")
        print(f"Nodes degree: {gra.degree}.")

    @timer
    def basic_heuristic_search(self):
        """Basic heuristic search for the problem.

        :return: (opt-round, (opt-partition-1, opt-partition-2)), last-time (exists with @timer)
        """

        part_1, part_2 = self.get_init_partition()
        c = -1
        while 1:
            c += 1
            node_pairs = [(n1, n2) for n1 in part_1 for n2 in part_2]

            cross_weight_dict = dict()
            for (n1, n2) in node_pairs:
                cross_weight_dict[(n1, n2)] = self._calculate_cross_partition_weights(n1, part_1, n2, part_2)
            best_key = min(cross_weight_dict, key=cross_weight_dict.get)
            value = cross_weight_dict[best_key]
            if value > 0:
                break
            n1, n2 = best_key
            self._exchange_node(n1, n2, part_1, part_2)

        return c, (part_1, part_2)

    @timer
    def tabu_heuristic_search(self, cmax=100, life=10):
        """Tabu heuristic search for the problem.

        :return: (opt-round, (opt-partition-1, opt-partition-2)), last-time (exists with @timer)
        """

        tabu_dict = dict()
        optimal_val = 1e+9
        optimal_partition = None
        optimal_c = -1

        part_1, part_2 = self.get_init_partition()

        for c in range(cmax):
            node_pairs = [(n1, n2) for n1 in part_1 for n2 in part_2]

            cross_weight_dict = dict()
            for (n1, n2) in node_pairs:
                if (n1, n2) in tabu_dict:
                    continue
                cross_weight_dict[(n1, n2)] = self._calculate_cross_partition_weights(n1, part_1, n2, part_2)

            best_key = min(cross_weight_dict, key=cross_weight_dict.get)
            value = cross_weight_dict[best_key]
            n1, n2 = best_key
            if value >= 0:
                tabu_dict[(n1, n2)] = life
                val = self._calculate_part_weights(part_1, part_2)
                if val < optimal_val:
                    optimal_val = val
                    optimal_partition = (part_1.copy(), part_2.copy())
                    optimal_c = c

            self._exchange_node(n1, n2, part_1, part_2)

            self.tabu_dict_update(tabu_dict)
        return optimal_c, optimal_partition

    @timer
    def simulated_annealing_search(self, cmax=100, T=100, alpha=0.999):
        """Simulated annealing heuristic search for the problem.

        :return: (opt-round, (opt-partition-1, opt-partition-2)), last-time (exists with @timer)
        """

        part_1, part_2 = self.get_init_partition()
        optimal_val = 1e+10
        optimal_partition = None
        optimal_c = -1

        for c in range(cmax):
            n1 = random.sample(part_1, 1)[0]
            n2 = random.sample(part_2, 1)[0]
            cross_weight = self._calculate_cross_partition_weights(n1, part_1, n2, part_2)
            if cross_weight < 0:
                self._exchange_node(n1, n2, part_1, part_2)
            else:
                val = self._calculate_part_weights(part_1, part_2)
                if val < optimal_val:
                    optimal_val = val
                    optimal_partition = (part_1.copy(), part_2.copy())
                    optimal_c = c

                prob = math.exp(-cross_weight / T)
                if random.random() < prob:
                    self._exchange_node(n1, n2, part_1, part_2)
            T = alpha * T

        return optimal_c, optimal_partition

    @staticmethod
    def _exchange_node(n1, n2, part_1, part_2):
        """Exchange node n1, n2 between part-1, part-2."""

        part_1.remove(n1)
        part_1.add(n2)
        part_2.remove(n2)
        part_2.add(n1)

    @staticmethod
    def tabu_dict_update(tabu_dict):
        """Update the tabu dict.

        :param tabu_dict: the tabu dictionary.
        """

        init_key = list(tabu_dict.keys())
        for key in init_key:
            if tabu_dict[key] == -1:
                del tabu_dict[key]
            else:
                tabu_dict[key] -= 1

    def calculate_partition_cross_weight(self, part_1: set, part_2: set):
        """External Function for get the sum of crossed weights."""
        return self._calculate_part_weights(part_1, part_2)

    def _calculate_part_weights(self, part_1: set, part_2: set):
        """Calculate the sum of crossed weights between two parts."""

        s = 0
        for n in part_1:
            _get_edge_weight = self._get_edge_weight_map_func(n)
            n_p2 = map(_get_edge_weight, part_2)
            s += sum(n_p2)
        return s

    def _calculate_cross_partition_weights(self, n1, part_1, n2, part_2):
        """Calculate the difference after and before the exchange of n1 and n2."""

        n1_p1, n1_p2 = self._get_edge_weights_node(n1, part_1, part_2)
        n2_p1, n2_p2 = self._get_edge_weights_node(n2, part_1, part_2)
        n1_n2 = self.graph.get_edge_data(n1, n2)['weight']
        return n1_p1 + n2_p2 - n1_p2 - n2_p1 + 2 * n1_n2

    def _get_edge_weights_node(self, n, part_1, part_2):
        """Get the sum of edges of node n to part-1 & part-2."""

        _get_edge_weight = self._get_edge_weight_map_func(n)
        n_p2 = map(_get_edge_weight, part_2)
        n_p1 = map(_get_edge_weight, part_1)

        return sum(n_p1), sum(n_p2)

    def _get_edge_weight_map_func(self, n):
        """Get the get edge weight function with fixed node n."""

        def _get_edge_weight(_n):
            edge_data = self.graph.get_edge_data(n, _n)
            if edge_data is None:
                return 0
            return edge_data['weight']

        return _get_edge_weight

    @staticmethod
    def _random_split_of_nodes(nodes):
        """Randomly split one node list into two partitions."""

        nodes = np.array(nodes)
        node_index = np.arange(len(nodes))
        np.random.shuffle(node_index)
        assert len(node_index) % 2 == 0
        middle = len(node_index) // 2
        part_1 = set(nodes[node_index[:middle]])
        part_2 = set(nodes[node_index[middle:]])
        return part_1, part_2


def output_result(gra: UniformGraphPartition, partition, tag: str = None, opt_round: int = None, time_last: float = 0.):
    """Transform the search partition result into the output form."""

    output_ = ""
    if tag is None:
        output_ += "*" * 4
    else:
        output_ += f"{tag}"
    output_ += " result:\n"
    output_ += f"* Optimal round: {opt_round}\n"
    output_ += "* Sum of crossed edge weight: {}\n".format(
        gra.calculate_partition_cross_weight(*partition))
    output_ += "* Algorithm time usage:{:.3f}s\n".format(time_last)
    output_ += f"* Part 1: {partition[0]}\n"
    output_ += f"* Part 2: {partition[1]}\n"

    return output_


def save_output(_output: str, path):
    """Save the output into the file."""

    split_str = '=' * 50 + '\n'
    with open(path, 'a') as file:
        file.write(split_str)
        file.write(_output)


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


if __name__ == '__main__':
    # parameters
    save_file_path = "./Outputs/result-uniform-graph-partition.txt"
    parameter_dict = {
        "graph_node_num_half": 50,
        "tabu_cmax": 100,
        "tabu_life": 10,
        "annealing_cmax": int(3e+4),
        "annealing_T": 1.5e+6,
        "annealing_alpha": 0.9994,
        "seed": 220506,
    }
    args = argparse.Namespace(**parameter_dict)
    print(args)
    # Save parameters.
    save_arg_parameters(args, save_file_path)

    # Set seed.
    set_seed(args.seed)

    # graph uniform partition module initialize.
    graph_uniform_partition = UniformGraphPartition(node_num_half=args.graph_node_num_half)
    # graph_uniform_partition.plot_graph()
    # graph_uniform_partition.output_graph()

    # ******************************************************************************************
    # Initial partition.
    init_partition = graph_uniform_partition.get_init_partition()
    output_init = output_result(graph_uniform_partition, init_partition, tag="Initial partition")
    print(output_init)
    graph_uniform_partition.plot_graph(pos_name='bipartite',
                                       first_part=init_partition[0],
                                       second_part=init_partition[1]
                                       )

    # ******************************************************************************************
    # Basic heuristic search.
    (basic_opt_c, basic_search), time_basic = graph_uniform_partition.basic_heuristic_search()
    output_basic = output_result(graph_uniform_partition, basic_search,
                                 tag="Basic search", opt_round=basic_opt_c, time_last=time_basic)
    print(output_basic)
    graph_uniform_partition.plot_graph(pos_name='bipartite',
                                       first_part=basic_search[0],
                                       second_part=basic_search[1]
                                       )

    # ******************************************************************************************
    # Tabu search.
    (tabu_opt_c, tabu_search), time_tabu = graph_uniform_partition.tabu_heuristic_search(
        cmax=args.tabu_cmax, life=args.tabu_life)
    output_tabu = output_result(graph_uniform_partition, tabu_search,
                                tag="Tabu search", opt_round=tabu_opt_c, time_last=time_tabu)
    print(output_tabu)
    graph_uniform_partition.plot_graph(pos_name='bipartite',
                                       first_part=tabu_search[0],
                                       second_part=tabu_search[1]
                                       )

    # ******************************************************************************************
    # Simulated annealing search.
    (annealing_opt_c, simulated_annealing), time_annealing = graph_uniform_partition.simulated_annealing_search(
        cmax=args.annealing_cmax, T=args.annealing_T, alpha=args.annealing_alpha)
    output_annealing = output_result(graph_uniform_partition, simulated_annealing,
                                     tag="Simulated annealing", opt_round=annealing_opt_c, time_last=time_annealing)
    print(output_annealing)
    graph_uniform_partition.plot_graph(pos_name='bipartite',
                                       first_part=simulated_annealing[0],
                                       second_part=simulated_annealing[1]
                                       )

    # Save results.
    for output in [output_init, output_basic, output_tabu, output_annealing]:
        save_output(output, save_file_path)
