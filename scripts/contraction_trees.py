#!/usr/bin/env python

import networkit as nk
import random, config
import pandas as pd
from anytree import Node, RenderTree, PreOrderIter, AnyNode


# g is graph on which the subgraph is build, s is a set of nodes
def subgraph(g, s):
    res = nk.Graph(len(s), weighted=g.isWeighted())
    node_ids = dict(zip(s, range(len(s))))
    for u in s:
        for n in g.neighbors(u):
            if u < n and n in s:
                res.addEdge(node_ids[u], node_ids[n], g.weight(u, n))

    return res

# For the creation of additional orderings one should rewrite the PreOrderIter function
def recursive_PLM(g):
    '''
    Recursively applies PLM to a graph until the partions are not greater than |g| * config.PLM_RESOLUTION.
    Returns the contraction tree. The leaves carry the old ids.
    '''

    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        i = 0
        before = pd.Timestamp.now()

    def rec_PLM(s, parent, old_node_ids):
        node = AnyNode(parent=parent)
        part = nk.community.PLM(s, par="none randomized").run().getPartition()
        if part.numberOfSubsets() == 1:
            AnyNode(parent=node, cluster=list(part.getMembers(0)), old_ids=[old_node_ids[i] for i in part.getMembers(0)])
            return

        for n in range(part.numberOfSubsets()):
            p = part.getMembers(n)
            if len(p) > g.numberOfNodes() * config.PLM_RESOLUTION:
                rec_PLM(subgraph(s, p), node, [old_node_ids[i] for i in p])
            else:
                AnyNode(parent=node, cluster=list(p), old_ids=[old_node_ids[i] for i in p])

    root = Node("root")
    rec_PLM(g, root, range(g.numberOfNodes()))
    #print(RenderTree(root))

    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        after = pd.Timestamp.now()
        print("Total time: {:f}s".format((after-before).total_seconds()))

    return root


# Needs an indexed graph
def set_weights_graph(g):
    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        before = pd.Timestamp.now()

    tes = nk.sparsification.TriangleEdgeScore(g)
    tes.run()
    scores = tes.scores()
    def set_weight_to_edge(u, v, weight, edgeid):
        # Divides common neighbors by the amount of total neighbors. Therefore the weight will be always < 1. u and v are counted as neighbors
        g.setWeight(u, v, (1 + scores[edgeid]) / (1 + len(g.neighbors(u)) + len(g.neighbors(v)) - scores[edgeid]))
    g.forEdges(set_weight_to_edge)

    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        after = pd.Timestamp.now()
        print("Setting weights: {:f}s".format((after-before).total_seconds()))

# each element is stored at the index that is double its value
class DisjointSet:
    '''
     Array based implementation of the disjoint set. Can only store integer values from 0 to n as elements,
     because they are used as indices in the array.
    '''

    # Stores the parent of element n at n*2 and its rank at n*2+1
    def __init__(self, amount_nodes):
        self._disjoint_set = [0] * (amount_nodes * 2)
        self.amount_nodes = amount_nodes
        for i in range(amount_nodes):
            self._disjoint_set[i*2] = i


    def _rank(self, node):
        return self._disjoint_set[node * 2 + 1]

    def _parent(self, node):
        return self._disjoint_set[node*2]

    def _set_parent(self, node, parent):
        self._disjoint_set[node*2] = parent

    def find(self, node):
        if node < 0 or node >= self.amount_nodes:
            raise ValueError("Node is not within range of the disjoint set")

        if self._parent(node) == node:
            return node
        else:
            self._set_parent(node, self.find(self._parent(node)))
            return self._parent(node)

    def union(self, node1, node2):
        if node1 < 0 or node1 >= self.amount_nodes:
            raise ValueError("Node 1 is not within range of the disjoint set")
        if node2 < 0 or node2 >= self.amount_nodes:
            raise ValueError("Node 2 is not within range of the disjoint set")

        root1 = self.find(node1)
        root2 = self.find(node2)
        if self._rank(root1) > self._rank(root2):
            self._set_parent(root2, root1)
        elif self._rank(root1) < self._rank(root2):
            self._set_parent(root1, root2)
        elif root1 != root2:
            self._set_parent(root2, root1)
            self._disjoint_set[root1*2+1] += 1 # increments the rank of root1

    # used for contractions and ordering
    # this usage is subject to change in future performance improvements
    # therefore this method might become unnecessary
    def get(self):
        d = dict()
        for node in range(self.amount_nodes):
            root = self.find(node)
            if root not in d:
                d[root] = [node]
            else:
                d.get(root).append(node)

        return d # before list(d.values())

# doesnt work for not connected graphs
def find_closest_neighbor_edges(g):
    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        before = pd.Timestamp.now()

    disjoint_set = DisjointSet(g.numberOfNodes())
    for u in g.iterNodes():
        strongest_neighbor = -1
        #Every neighbor has a weight >= 0
        strongest_weight = 0.0
        for n in g.neighbors(u):
            if g.weight(u, n) >= strongest_weight:
                strongest_neighbor = n
                strongest_weight = g.weight(u, n)
        if strongest_neighbor > -1:
            disjoint_set.union(u, strongest_neighbor)

    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        after = pd.Timestamp.now()
        print("Finding strongest edges: {:f}s".format((after-before).total_seconds()))

    return disjoint_set


# treats edges with weight 0 and non connected vertices equally
def contract_to_nodes(g, disjoint_set, d):
    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        before = pd.Timestamp.now()

    # overhead shouldnt be too much.
    # there hopefully is a cleaner solution
    node_to_rep = list()
    for rep in d:
        node_to_rep.append(rep)

    rep_to_node = dict()
    cnt = 0
    for rep in d:
        rep_to_node[rep] = cnt
        cnt += 1

    contracted_g = nk.Graph(len(d), weighted=True)
    contracted_g.indexEdges()

    def sumWeights(u, v, weight, edgeid):
        if disjoint_set.find(u) != disjoint_set.find(v):
            contracted_g.increaseWeight(rep_to_node[disjoint_set.find(u)], rep_to_node[disjoint_set.find(v)], g.weight(u, v))

    g.forEdges(sumWeights)

    def buildAvgWeights(u, v, weight, edgeid):
        contracted_g.setWeight(u, v, weight / (len(d[node_to_rep[u]]) * len(d[node_to_rep[v]])))

    contracted_g.forEdges(buildAvgWeights)

    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        after = pd.Timestamp.now()
        print("Contracting graph: {:f}s".format((after-before).total_seconds()))

    return contracted_g

def affinity_tree(g):
    '''
    Creates a contraction tree as seen in Aydin, K.; Bateni, M.H.; Mirrokni, V.: Distributed Balanced Partitioning via Linear Embedding.

    '''
    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        i = 0
    set_weights_graph(g)

    curr_contraction = find_closest_neighbor_edges(g)
    dict_contraction = curr_contraction.get()
    g = contract_to_nodes(g, curr_contraction, dict_contraction)

    last_iteration = list()
    for k, v in dict_contraction.items():
        last_iteration.append(AnyNode(old_ids=v))

    # officially supported python construct for 'do ... while'
    while True:
        if config.TIME_STAMPS >= config.TimeStamps.ALL:
            print("------ Layer {:d} ------".format(i))
            i += 1

        curr_contraction = find_closest_neighbor_edges(g)
        dict_contraction = curr_contraction.get()
        g = contract_to_nodes(g, curr_contraction, dict_contraction)

        curr_iteration = list()
        for k, v in dict_contraction.items(): # Should be the order of the contractions
            n = AnyNode()
            n.children = [last_iteration[i] for i in v]
            curr_iteration.append(n)

        last_iteration = curr_iteration

        if len(dict_contraction) == 1:
            break;

    return last_iteration[0]


# def main():
#     '''
#     Quick test method.
#     '''
#
#     g = load_graph("../affinity/walshaw/uk.graph", nk.graphio.METISGraphReader())
#
#     # Test to check for not connected graphs. Remove for more performance
#     comp = nk.components.ConnectedComponents(g)
#     comp.run()
#     if comp.numberOfComponents() > 1:
#         return
#
#     root = affinity_tree(g)
#
#     orders = get_orderings(root, 3, connected_random_reorder_func(g))


if __name__ == "__main__":
    main()
