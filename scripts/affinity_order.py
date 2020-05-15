#!/usr/bin/env python

import networkit as nk
import itertools, random, config
import pandas as pd
from anytree import Node, RenderTree, PreOrderIter, AnyNode
from anytree.iterators import AbstractIter
from collections import Counter

def calculate_and_save_order(path_to_graph, path_to_ord, ordering_alg, amount_orders, reader=nk.graphio.METISGraphReader()):
    g = load_graph(path_to_graph, reader)

    # Test to check for not connected graphs. Remove for more performance
    comp = nk.components.ConnectedComponents(g)
    comp.run()
    if comp.numberOfComponents() > 1:
        return

    orders = ordering_alg(g, amount_orders)
    with open(path_to_ord, "w") as f:
        f.write(config.DELIMITER_ORDER.join(config.DELIMITER_NODE.join(str(i) for i in order) for order in orders))

def load_graph(path_to_graph, reader=nk.graphio.METISGraphReader()):
    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        before = pd.Timestamp.now()

    g = nk.graphtools.toWeighted(reader.read(path_to_graph))
    g.indexEdges()

    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        after = pd.Timestamp.now()
        print("Loading graph: {:f}s".format((after-before).total_seconds()))

    return g

def random_reorder(l):
    return random.sample(l, len(l))

# Returns a function that realizes a randomized-connected reordering on the given graph g
def connected_random_reorder_func(g):

    def connected_random_reorder(l):
        if len(l) <= 1:
            return l

        nodes = dict()
        for cluster in l:
            nodes[cluster] = set([node_id for node in PreOrderIter(cluster) if node.is_leaf for node_id in node.old_ids])

        #choose random cluster c
        c = random.choice(l)
        added_clusters = list()
        added_clusters.append(c)

        #create list with #edges to the cluster
        edge_count = Counter()

        while True:
            del edge_count[c] # stops the code from adding the same cluster twice

            for u in nodes[c]:
                for n in g.neighbors(u):
                    # should run in O(1)
                    # finds the cluster that n is in and incerments its edge count
                    for cluster in l:
                        if cluster in added_clusters: # assumes that len(l) is small
                            continue

                        if n in nodes[cluster]:
                            edge_count[cluster] += 1
                            break

            c = edge_count.most_common(1)[0][0] # list of tuples
            added_clusters.append(c)
            if len(added_clusters) >= len(l):
                break

        return added_clusters

    return connected_random_reorder



def recursive_PLM_orderings(g, amount_orders):
    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        i = 0
        before = pd.Timestamp.now()

    root = recursive_PLM(g)

    orderings = get_orderings(root, amount_orders, connected_random_reorder_func(g))

    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        after = pd.Timestamp.now()
        print("Total time: {:f}s".format((after-before).total_seconds()))

    return orderings

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

def get_ordering_PLM(root, reorder):
    # Slightly modified from https://github.com/c0fec0de/anytree/blob/master/anytree/iterators/preorderiter.py
    class RandomDFSIter(AbstractIter):

        @staticmethod
        def _iter(children, filter_, stop, maxlevel):
            for child_ in reorder(children):
                if stop(child_):
                    continue
                if filter_(child_):
                    yield child_
                if not AbstractIter._abort_at_level(2, maxlevel):
                    descendantmaxlevel = maxlevel - 1 if maxlevel else None
                    for descendant_ in RandomDFSIter._iter(child_.children, filter_, stop, descendantmaxlevel):
                        yield descendant_

    # Creates a flat list containing the ordering. Only leafs have a old_ids attribute
    ordering = [node_id for node in RandomDFSIter(root) if node.is_leaf for node_id in node.old_ids]

    return ordering

def get_orderings(root, n, reorder=None, reorders=None):
    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        before = pd.Timestamp.now()

    if reorder == None and reorders == None:
        raise AttributeError("At least one reorder function must be given")

    random.seed(config.SEED)

    def get_orderings_from_tree(root, reorders):
        ret = list()
        for i in range(n):
            ret.append(get_ordering_PLM(root, reorders[i]))

        return ret

    if reorder != None:
        ret = get_orderings_from_tree(root, [reorder]*n)
    else:
        if len(reorders) != n:
            raise ValueError("The amount of reorder functions must match the amount of orders")
        ret = get_orderings_from_tree(root, reorders)

    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        after = pd.Timestamp.now()
        print("Calculating orders: {:f}s".format((after-before).total_seconds()))

    return ret

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

def affinity_orderings(g, amount_orders):
    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        i = 0
        before = pd.Timestamp.now()

    root = affinity_tree(g)

    orderings = get_orderings(root, amount_orders, connected_random_reorder_func(g))

    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        after = pd.Timestamp.now()
        print("Total time: {:f}s".format((after-before).total_seconds()))

    return orderings

ORD_ALG = dict(zip(config.ORD_TYPE, [recursive_PLM_orderings, affinity_orderings]))

def main():
    g = load_graph("../affinity/walshaw/uk.graph", nk.graphio.METISGraphReader())

    # Test to check for not connected graphs. Remove for more performance
    comp = nk.components.ConnectedComponents(g)
    comp.run()
    if comp.numberOfComponents() > 1:
        return

    root = affinity_tree(g)

    orders = get_orderings(root, 3, connected_random_reorder_func(g))


if __name__ == "__main__":
    main()
