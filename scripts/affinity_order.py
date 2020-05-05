#!/usr/bin/env python

import networkit as nk
import itertools, pandas, random
from anytree import Node, RenderTree, PreOrderIter, AnyNode
from anytree.iterators import AbstractIter


# Makes the output more verbose and adds timings
# Maybe represent as enum in the future for more granular control
TIME_STAMPS = False
DELIMITER_NODE = ","
DELIMITER_ORDER = "\n"
RESOLUTION = 0.01

def main():
    TIME_STAMPS = True
    WALSHAW_GRAPH = True
    AMOUNT_ORDERS = 10

    GRAPH = "fe_ocean"
    GRAPH_ENDING = ".graph"
    WALSHAW_PATH = "../affinity/walshaw/"

    ORD_PATH = "../affinity/orders/"
    ORD_END = "-aff.ord"
    path_to_ord = ORD_PATH + GRAPH + ORD_END

    if WALSHAW_GRAPH:
        path_to_graph = WALSHAW_PATH + GRAPH + GRAPH_ENDING
    else:
        path_to_graph = GRAPH + GRAPH_ENDING

    calculate_and_save_order(path_to_graph, path_to_ord, AMOUNT_ORDERS)

def calculate_and_save_order(path_to_graph, path_to_ord, ordering_alg, amount_orders, reader=nk.graphio.METISGraphReader()):
    g = load_graph(path_to_graph, reader)

    # Test to check for not connected graphs. Remove for more performance
    comp = nk.components.ConnectedComponents(g)
    comp.run()
    if comp.numberOfComponents() > 1:
        return

    orders = ordering_alg(g, amount_orders)
    with open(path_to_ord, "w") as f:
        f.write(DELIMITER_ORDER.join(DELIMITER_NODE.join(str(i) for i in order) for order in orders))

def load_graph(path_to_graph, reader):
    if TIME_STAMPS:
        before = pandas.Timestamp.now()

    g = nk.graphtools.toWeighted(reader.read(path_to_graph))
    g.indexEdges()

    if TIME_STAMPS:
        after = pandas.Timestamp.now()
        print("Loading graph: {:f}s".format((after-before).total_seconds()))

    return g

# TODO: implement for multiple orders
def recursive_PLM_orderings(g, amount_orders):
    if TIME_STAMPS:
        i = 0
        before = pandas.Timestamp.now()

    root = recursive_PLM(g)

    orderings = get_orderings_PLM(root, amount_orders, random_reorder)

    if TIME_STAMPS:
        after = pandas.Timestamp.now()
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

    if TIME_STAMPS:
        i = 0
        before = pandas.Timestamp.now()

    def rec_PLM(s, parent, old_node_ids):
        node = AnyNode(parent=parent)
        part = nk.community.PLM(s, par="none randomized").run().getPartition()
        if part.numberOfSubsets() == 1:
            AnyNode(parent=node, cluster=list(part.getMembers(0)), old_ids=[old_node_ids[i] for i in part.getMembers(0)])
            return

        for n in range(part.numberOfSubsets()):
            p = part.getMembers(n)
            if len(p) > g.numberOfNodes() * RESOLUTION:
                rec_PLM(subgraph(s, p), node, [old_node_ids[i] for i in p])
            else:
                AnyNode(parent=node, cluster=list(p), old_ids=[old_node_ids[i] for i in p])

    root = Node("root")
    rec_PLM(g, root, range(g.numberOfNodes()))
    #print(RenderTree(root))

    if TIME_STAMPS:
        after = pandas.Timestamp.now()
        print("Total time: {:f}s".format((after-before).total_seconds()))

    return root

def get_ordering_PLM(root, reorder):
    class DFSIter(AbstractIter):

        @staticmethod
        def _iter(children, filter_, stop, maxlevel):
            for child_ in reorder(children):
                if stop(child_):
                    continue
                if filter_(child_):
                    yield child_
                if not AbstractIter._abort_at_level(2, maxlevel):
                    descendantmaxlevel = maxlevel - 1 if maxlevel else None
                    for descendant_ in PreOrderIter._iter(child_.children, filter_, stop, descendantmaxlevel):
                        yield descendant_

    # Creates a flat list containing the ordering. Only leafs have a old_ids attribute
    ordering = [node_id for node in DFSIter(root) if node.is_leaf for node_id in node.old_ids]

    return ordering

# Basically the same as get_orderings
def get_orderings_PLM(root, n, reorder=None, reorders=None):
    if TIME_STAMPS:
        before = pandas.Timestamp.now()

    if reorder == None and reorders == None:
        raise AttributeError("At least one reorder function must be given")

    def get_orderings_from_contractions(list_contractions, n, reorders):
        ret = list()
        for i in range(n):
            ret.append(get_ordering_PLM(list_contractions, reorders[i]))

        return ret

    if reorder != None:
        ret = get_orderings_from_contractions(root, n, [reorder]*n)
    else:
        if len(reorders) != n:
            raise ValueError("The amount of reorder functions must match the amount of orders")
        ret = get_orderings_from_contractions(root, n, reorders)

    if TIME_STAMPS:
        after = pandas.Timestamp.now()
        print("Calculating orders: {:f}s".format((after-before).total_seconds()))

    return ret

# Needs an indexed graph
def set_weights_graph(g):
    if TIME_STAMPS:
        before = pandas.Timestamp.now()

    tes = nk.sparsification.TriangleEdgeScore(g)
    tes.run()
    scores = tes.scores()
    def set_weight_to_edge(u, v, weight, edgeid):
        # Divides common neighbors by the amount of total neighbors. Therefore the weight will be always < 1. u and v are counted as neighbors
        g.setWeight(u, v, (1 + scores[edgeid]) / (1 + len(g.neighbors(u)) + len(g.neighbors(v)) - scores[edgeid]))
    g.forEdges(set_weight_to_edge)

    if TIME_STAMPS:
        after = pandas.Timestamp.now()
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
    if TIME_STAMPS:
        before = pandas.Timestamp.now()

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

    if TIME_STAMPS:
        after = pandas.Timestamp.now()
        print("Finding strongest edges: {:f}s".format((after-before).total_seconds()))

    return disjoint_set


# treats edges with weight 0 and non connected vertices equally
def contract_to_nodes(g, disjoint_set, d):
    if TIME_STAMPS:
        before = pandas.Timestamp.now()

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

    if TIME_STAMPS:
        after = pandas.Timestamp.now()
        print("Contracting graph: {:f}s".format((after-before).total_seconds()))

    return contracted_g

def get_ordering_from_contractions(list_contractions, reorder):
    def rec_ordering(layer, index):
        if layer > 0:
            ordering = list()
            for i in reorder(list_contractions[layer][index]):
                ordering += rec_ordering(layer-1, i)

            return ordering
        else:
            return list_contractions[layer][index]

    ordering = rec_ordering(len(list_contractions)-1, 0)
    return ordering

# Calculates  multiple contractions from the given reorderings
def get_orderings(list_contractions, n, reorder=None, reorders=None):
    if TIME_STAMPS:
        before = pandas.Timestamp.now()

    if reorder == None and reorders == None:
        raise AttributeError("At least one reorder function must be given")

    def get_orderings_from_contractions(list_contractions, n, reorders):
        ret = list()
        for i in range(n):
            ret.append(get_ordering_from_contractions(list_contractions, reorders[i]))

        return ret

    if reorder != None:
        ret = get_orderings_from_contractions(list_contractions, n, [reorder]*n)
    else:
        if len(reorders) != n:
            raise ValueError("The amount of reorder functions must match the amount of orders")
        ret = get_orderings_from_contractions(list_contractions, n, reorders)

    if TIME_STAMPS:
        after = pandas.Timestamp.now()
        print("Calculating orders: {:f}s".format((after-before).total_seconds()))

    return ret

def random_reorder(l):
    return random.sample(l, len(l))

def affinity_orderings(g, amount_orders):
    if TIME_STAMPS:
        i = 0
        before = pandas.Timestamp.now()

    set_weights_graph(g)
    contractions = list()

    # officially supported python construct for 'do ... while'
    while True:
        if TIME_STAMPS:
            print("------ Layer {:d} ------".format(i))
            i += 1

        curr_contraction = find_closest_neighbor_edges(g)
        dict_contraction = curr_contraction.get()
        g = contract_to_nodes(g, curr_contraction, dict_contraction)
        contractions.append(list(dict_contraction.values()))

        if len(curr_contraction.get()) == 1:
            break;

    orderings = get_orderings(contractions, amount_orders, random_reorder)

    if TIME_STAMPS:
        after = pandas.Timestamp.now()
        print("Total time: {:f}s".format((after-before).total_seconds()))

    return orderings

if __name__ == '__main__':
    main()
