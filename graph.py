#######################################################################
# Name: env.py
# - Simple graph class with utility functions.
# - Adapted from https://gist.github.com/betandr/541a1f6466b6855471de5ca30b74cb31
#######################################################################

import sys
if sys.modules['TRAINING']:
    from parameter import *
else:
    from test_parameter import *

import numpy as np
from collections import deque


class Edge:
    def __init__(self, to_node, length):
        self.to_node = to_node
        self.length = length


class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = dict()

    def add_node(self, node):
        """ Add node to graph """
        self.nodes.add(node)

    def add_edge(self, from_node, to_node, length):
        """ Add edge to graph """
        edge = Edge(to_node, length)
        if from_node in self.edges:
            from_node_edges = self.edges[from_node]
        else:
            self.edges[from_node] = dict()
            from_node_edges = self.edges[from_node]

        from_node_edges[to_node] = edge

    def clear_edge(self, from_node, remove_bidirectional_edges=False):
        """ Clear edge from graph """
        if remove_bidirectional_edges:
            for to_node in list(self.edges.keys()):
                if from_node in self.edges[to_node]:
                    del self.edges[to_node][from_node]
        
        if from_node in self.edges:
            self.edges[from_node] = dict()

    def clear_node(self, node, remove_bidirectional_edges=False):
        """ Clear node from graph """
        self.clear_edge(node, remove_bidirectional_edges=remove_bidirectional_edges)

        # Remove the node from the set of nodes
        if node in self.nodes:
            self.nodes.remove(node)


    def is_connected_bfs(self, start_node, criteria=None):
        """Check if the graph is connected using BFS."""

        # An empty graph is considered connected
        if len(self.nodes) == 0:
            return True  

        visited = set()
        queue = deque([start_node])

        while len(queue) != 0:
            current_node = queue.popleft()
            if current_node not in visited:
                visited.add(current_node)
                neighbor_nodes = set(self.edges[current_node].keys())
                queue.extend(neighbor_nodes - visited)

        # To make sure edges cleared properly
        visited = set(map(tuple, visited)).intersection(set(map(tuple, self.nodes)))
        if criteria is not None:
            criteria_bounded = criteria.intersection(set(map(tuple, self.nodes)))

        if criteria is None:
            is_connected = (len(visited) == len(self.nodes))
        else:
            is_connected = criteria_bounded.issubset(visited)
        return is_connected, list(visited)



def h(index, destination):
    """ h function for a-star """
    current = np.array(index)
    end = np.array(destination)
    h = np.linalg.norm(end-current)
    return h


def a_star(start, destination, graph):    
    """ A-star path planning algorithm """
    if start == destination:
        return [], 0, set([]), ([],[])
    if tuple(destination) in graph.edges[tuple(start)].keys():
        cost = graph.edges[tuple(start)][tuple(destination)].length
        return [start, destination], cost, set([]), ([],[])
    open_list = {start}
    closed_list = set([])
    edges_explored_list = ([],[])

    g = {start: 0}
    parents = {start: start}

    while len(open_list) > 0:       
        n = None        # current node with lowest f cost
        h_n = 1e5

        # Choose vertex with next lowest f cost
        for v in open_list:
            h_v = h(v, destination)
            if n is not None:
                h_n = h(n, destination)
            if n is None or g[v] + h_v < g[n] + h_n:
                n = v

        if n is None:
            # print('[1] Path does not exist!')
            return None, 1e5, closed_list, edges_explored_list

        # If found destination, backtrack to generate astar path
        if n == destination:
            reconst_path = []
            while parents[n] != n:
                reconst_path.append(n)
                n = parents[n]
            reconst_path.append(start)
            reconst_path.reverse()
            return reconst_path, g[destination], closed_list, edges_explored_list

        for edge in graph.edges[tuple(n)].values():

            m = tuple(edge.to_node)
            edges_explored_list[0].append(n)
            edges_explored_list[1].append(m)
            cost = edge.length

            if m in closed_list:
                continue

            if m not in open_list:
                open_list.add(m)
                parents[m] = n
                g[m] = g[n] + cost

            elif g[m] > g[n] + cost:
                parents[m] = n
                g[m] = g[n] + cost

        open_list.remove(n)
        closed_list.add(n)
    
    # print('[2] Path does not exist!')
    return None, 1e5, closed_list, edges_explored_list