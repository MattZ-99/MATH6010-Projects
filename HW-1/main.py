import networkx as nx
from matplotlib import pyplot as plt
from networkx.generators.random_graphs import fast_gnp_random_graph
import random
import math


class SolutionForDominatingSet(object):
    """ Solution for MATH6010 Homework-1 """

    def __init__(self):
        super(SolutionForDominatingSet, self).__init__()
        self.graph = nx.Graph()
        self.min_delta = 0
        self.dominating_set = None
        self.select_list = None

    def random_generate_graph(self, node_num=10, p=0.3, delta=2) -> bool:
        """
        :param node_num: Number of generating node number.
        :param p: Probability for binomial generator.
        :param delta: Minimum degree delta > 1
        :return: Bool. True: generate success. False: generate failure
        """

        for _i in range(1000):
            self.graph.clear()
            self.graph = fast_gnp_random_graph(n=node_num, p=p)
            self.min_delta = min([d for n, d in self.graph.degree])
            if self.min_delta >= delta:
                return True

        self.graph.clear()
        self.__init__()
        return False

    def plot_graph(self) -> None:
        """
        Plot self.graph

        :return: None
        """

        nx.draw(self.graph, with_labels=True)
        plt.show()
        plt.close('all')

    def plot_graph_with_dominating_set(self):
        """
        Plot self.graph. Dominating set will be set to red, if existing.

        :return: None
        """

        if self.dominating_set is None:
            self.plot_graph()
            return
        node_list = list(self.graph.nodes)
        node_color = list()
        for node in node_list:
            if node in self.dominating_set:
                node_color.append("tab:red")
            else:
                node_color.append("tab:blue")
        nx.draw_networkx(self.graph, with_labels=True, nodelist=node_list, node_color=node_color)
        plt.show()
        plt.close('all')

    def _greedy_select_one_node(self, dominated_set: set, undominated_set: set) -> int:
        """
        Select a node for dominating set, using greedy algorithm.

        :param dominated_set: set
        :param undominated_set: set
        :return: selected_node: int
        """

        candidate_set = dominated_set | undominated_set
        degree_modified_set = set()
        for candidate_node in candidate_set:
            neighbours_modified = set(self.graph.neighbors(candidate_node))
            neighbours_modified.add(candidate_node)
            neighbours_modified = neighbours_modified & undominated_set

            degree_modified_set.add((candidate_node, len(neighbours_modified)))
        return max(degree_modified_set, key=lambda t: t[1])[0]

    def greedy_for_dominating_set(self):
        """
        Select dominating set, using greedy set.

        :return: None
        """

        dominating_set = set()
        dominated_set = set()
        undominated_set = set(self.graph.nodes)
        select_list = list()

        for _i in range(len(self.graph)):
            if len(undominated_set) == 0:
                break

            selected_node = self._greedy_select_one_node(dominated_set, undominated_set)

            dominating_set.add(selected_node)
            dominated_set = dominated_set | set(self.graph.neighbors(selected_node)) - dominating_set
            if selected_node in undominated_set:
                undominated_set.remove(selected_node)
            undominated_set -= set(self.graph.neighbors(selected_node))
            select_list.append(selected_node)
        self.select_list = select_list

        self.dominating_set = dominating_set


if __name__ == '__main__':
    solution = SolutionForDominatingSet()

    counter = 0
    while counter < 1000:
        if solution.random_generate_graph(node_num=100, p=random.uniform(a=0.2, b=0.6)):
            # print(solution.min_delta)
            if solution.min_delta == 100 * 0.2:
                # solution.plot_graph()
                solution.greedy_for_dominating_set()
                # print(dominating_set)
                # solution.plot_graph_with_dominating_set()
                print(counter, solution.select_list, len(solution.select_list), 100*(1 + math.log(100 * 0.2 + 1))/(100 * 0.2 + 1))

                counter += 1

