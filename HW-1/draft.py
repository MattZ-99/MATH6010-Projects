from networkx.generators.random_graphs import erdos_renyi_graph
import networkx as nx
import matplotlib.pyplot as plt

from collections import defaultdict
import networkx as nx


class Graph(object):
    """ Graph data structure, undirected by default. """

    def __init__(self, connections=None, directed=False):
        self._graph = defaultdict(set)
        self._directed = directed
        self.add_connections(connections)

    def add_connections(self, connections=None) -> None:
        """ Add connections (list of tuple pairs) to graph """

        if connections is None:
            return
        for node1, node2 in connections:
            self.add(node1, node2)

    def add(self, node1, node2) -> bool:
        """ Add connection between node1 and node2 """

        self._graph[node1].add(node2)
        if not self._directed:
            self._graph[node2].add(node1)
        return True

    def remove(self, node):
        """ Remove all references to node """

        for n, edge in self._graph.items():
            edge.discard(n)
        try:
            del self._graph[node]
        except KeyError:
            pass

    def clear(self):
        """ Clear the whole graph """

        self._graph.clear()

    def is_connected(self, node1, node2):
        """ Is node1 directly connected to node2 """

        return node1 in self._graph and node2 in self._graph[node1]

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, dict(self._graph))


n = 100
p = 0.1
g = erdos_renyi_graph(n, p)

# print(g.nodes)
# print(g.edges)
# print(g.degree)
degree = [d for n, d in g.degree]
print(degree)
print(min(degree))

nx.draw(g, with_labels=True)
plt.show()
