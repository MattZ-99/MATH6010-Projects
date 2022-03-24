import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from networkx.generators.random_graphs import fast_gnp_random_graph
import random
import math
from matplotlib.ticker import MaxNLocator
import os

# plt.rcParams['text.usetex'] = True


class SolutionForDominatingSet(object):
    """ Solution for MATH6010 Homework-1 """

    def __init__(self):
        super(SolutionForDominatingSet, self).__init__()
        self.graph = nx.Graph()
        self.min_delta = 0
        self.dominating_set = None
        self.select_list = None
        self.pos = None

    def random_generate_graph(self, node_num=10, p=0.3, delta=2) -> bool:
        """
        :param node_num: Number of generating node number.
        :param p: Probability for binomial generator.
        :param delta: Minimum degree delta > 1
        :return: Bool. True: generate success. False: generate failure
        """

        for _i in range(2000):
            self.graph.clear()
            self.graph = fast_gnp_random_graph(n=node_num, p=p)
            self.min_delta = min([d for n, d in self.graph.degree])
            if self.min_delta >= delta:
                self.pos = nx.spring_layout(self.graph)
                return True

        self.graph.clear()
        self.__init__()
        return False

    def plot_graph(self, path='./graph.jpg') -> None:
        """
        Plot self.graph

        :return: None
        """

        fig = plt.figure()
        plt.title(fr"Vertices num: {self.graph.number_of_nodes()}. Minimum degree $\delta$={self.min_delta}")
        nx.draw(self.graph, with_labels=True, pos=self.pos)
        # plt.show()
        plt.savefig(path)
        plt.close('all')

    def plot_graph_with_dominating_set(self, path='./graph_with_dominating_set.jpg'):
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

        fig = plt.figure()
        str_title = fr"Vertices num: {self.graph.number_of_nodes()}. Minimum degree $\delta$={self.min_delta}"
        str_title += "\nDominating set with order: "
        str_title += str(self.select_list)
        plt.title(str_title)
        nx.draw_networkx(self.graph, pos=self.pos, with_labels=True, nodelist=node_list, node_color=node_color)
        plt.axis('off')
        # plt.show()
        plt.savefig(path)
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


def makedirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def run_plot_generated_graph(s: SolutionForDominatingSet, save_dir: str = "./", num: int = 10, delta: int = 2) -> bool:
    """
    Run demo for generating graph, greedy algorithm for dominating set, and plot graph.
    :param s: SolutionForDominatingSet
    :param save_dir: str. Save dir.
    :param num: int. Node numbers.
    :param delta: int. Minimum degree. delta>1.
    :return: bool. Whether running success.
    """
    print(f"node number: {num}, delta: {delta}. "
          f"Generating graph, calculating dominating set, and plotting graph. Running...")

    for __try_times in range(100):
        if solution.random_generate_graph(node_num=num, p=random.uniform(a=0.2, b=0.6), delta=delta):
            s.plot_graph(path=os.path.join(save_dir, f"graph_node-num_{num}_delta_{delta}.png"))
            s.greedy_for_dominating_set()
            s.plot_graph_with_dominating_set(
                path=os.path.join(save_dir, f"graph-with-dominating-set_node-num_{num}_delta_{delta}.png")
            )
            return True
    return False


def run_stat_steps_for_dominating_graph(s: SolutionForDominatingSet, save_dir: str = "./",
                                        num: int = 100, delta: int = 20, count_num=100) -> bool:
    """
    Make statistics for steps of greedy algorithm.
    :param s: SolutionForDominatingSet
    :param save_dir: str.
    :param num: int.
    :param delta: int.
    :param count_num: count_num.
    :return: bool.
    """

    print(f"node number: {num}, delta: {delta}. Making statistics for steps of greedy algorithm. Running...")

    counter = 0
    step_list = list()
    while counter < count_num:
        if s.random_generate_graph(node_num=num,
                                   p=random.uniform(max(1. * delta / num-0.2, 0), min(1. * delta / num+0.2, 1))):
            if s.min_delta == delta:
                s.greedy_for_dominating_set()
                step_list.append(len(s.select_list))
                counter += 1
                # print(counter, 1. * delta / num)
            # print(s.min_delta)
    theory_limit = num * (1 + math.log(delta + 1)) / (delta + 1)

    fig = plt.figure()
    plt.title(fr"Vertices num: {num}. Minimum degree $\delta$={delta}")
    bins = [i - 0.5 for i in range(min(step_list), max(step_list) + 2, 1)]
    freq, bins, patches = plt.hist(step_list, bins=bins)

    # x coordinate for labels
    bin_centers = np.diff(bins) * 0.5 + bins[:-1]

    _n = 0
    for fr, x, patch in zip(freq, bin_centers, patches):
        height = int(freq[_n])
        plt.annotate("{}".format(height),
                     xy=(x, height),  # top left corner of the histogram bar
                     xytext=(0, 0.2),  # offsetting label position above its bar
                     textcoords="offset points",  # Offset (in points) from the *xy* value
                     ha='center', va='bottom'
                     )
        _n = _n + 1
    
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.axvline(x=theory_limit, color='red', linestyle='--')
    
    plt.xlabel("Size of dominating set")
    plt.ylabel("Number of occurrences")
    plt.annotate(r"$n[1+\ln(\delta + 1)]/(\delta + 1)$"+'\n'+'={:.2f}'.format(num*(1+math.log(delta+1))/(delta+1)),
                 xy=(num*(1+math.log(delta+1))/(delta+1), 0),
                 xytext=(-60, -25),
                 textcoords="offset points",
                 )
    
    # plt.show()
    plt.savefig(os.path.join(save_dir, f'steps_hist_node-num_{num}_delta_{delta}.png'))
    plt.close('all')

    return True


if __name__ == '__main__':
    output_dir = "./Outputs"
    output_graph_dir = os.path.join(output_dir, "GraphVisualization")
    output_stat_hist_dir = os.path.join(output_dir, "StatHist")
    makedirs(output_graph_dir)
    makedirs(output_stat_hist_dir)

    solution = SolutionForDominatingSet()

    run_plot_generated_graph(solution, output_graph_dir, num=4, delta=2)
    run_plot_generated_graph(solution, output_graph_dir, num=6, delta=2)
    run_plot_generated_graph(solution, output_graph_dir, num=10, delta=2)
    run_plot_generated_graph(solution, output_graph_dir, num=10, delta=4)

    run_stat_steps_for_dominating_graph(solution, output_stat_hist_dir, num=4, delta=2, count_num=50)
    run_stat_steps_for_dominating_graph(solution, output_stat_hist_dir, num=6, delta=2, count_num=50)
    run_stat_steps_for_dominating_graph(solution, output_stat_hist_dir, num=10, delta=2, count_num=50)
    run_stat_steps_for_dominating_graph(solution, output_stat_hist_dir, num=40, delta=10, count_num=50)
    run_stat_steps_for_dominating_graph(solution, output_stat_hist_dir, num=100, delta=10, count_num=50)
    # run_stat_steps_for_dominating_graph(solution, output_stat_hist_dir, num=100, delta=25, count_num=50)
    # run_stat_steps_for_dominating_graph(solution, output_stat_hist_dir, num=100, delta=50, count_num=50)
    # run_stat_steps_for_dominating_graph(solution, output_stat_hist_dir, num=200, delta=20, count_num=50)



