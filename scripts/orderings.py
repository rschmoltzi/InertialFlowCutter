import networkit as nk
import contraction_trees, algebraic_distances, config, random
from anytree import Node, RenderTree, PreOrderIter, AnyNode
from anytree.iterators import AbstractIter
from collections import Counter
import pandas as pd

'''
Top level ordering abstraction.
'''

def load_graph(path_to_graph, reader=nk.graphio.METISGraphReader()):
    '''
    Loads the graph specified by the path. For graph formats other than METIS a fiiting networkit reader must be supplied.
    '''

    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        before = pd.Timestamp.now()

    g = nk.graphtools.toWeighted(reader.read(path_to_graph))
    g.indexEdges()

    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        after = pd.Timestamp.now()
        print("Loading graph: {:f}s".format((after-before).total_seconds()))

    return g

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


def get_ordering(root, reorder):
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
    '''
    Generates multiple DFS based pseudo random orderings from the given contraction tree. Ordering[x] == y means that
    node y has the position x in the ordering. The pseudo randomness is based on config.SEED. This method
    uses the reorder argument to reorder the evaluation of all orderings, if given.
    Otherwise the reorders argument is used to supply a reorder method for each ordering.
    '''

    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        before = pd.Timestamp.now()

    if reorder == None and reorders == None:
        raise AttributeError("At least one reorder function must be given")

    random.seed(config.SEED)

    def get_orderings_from_tree(root, reorders):
        ret = list()
        for i in range(n):
            ret.append(get_ordering(root, reorders[i]))

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

def random_reorder(l):
    return random.sample(l, len(l))

def id(l):
    return l

# Returns a function that realizes a randomized-connected reordering on the given graph g
def connected_random_reorder_func(g):
    '''
    Wrapper function used to put g in the namespace of the returned functino without changing its signature.
    Returns a function that pseudo randomly reorders a list of clusters based on their connectivity in g.
    '''

    def connected_random_reorder(l):
        if len(l) <= 1:
            return l

        nodes = dict()
        for cluster in l:
            nodes[cluster] = set([node_id for node in PreOrderIter(cluster) if node.is_leaf for node_id in node.old_ids])

        # inverse map
        # for c in nodes:
        #     for u in

        #choose random cluster c
        c = random.choice(l)
        # ordered dict
        added_clusters = list()
        added_clusters.append(c)

        #create list with #edges to the cluster
        edge_count = Counter()

        while True:
            del edge_count[c] # stops the code from adding the same cluster twice

            for u in nodes[c]:
                for n in g.neighbors(u):
                    # Tidy this up with inverse map
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

def recursive_PLM_orderings(g, amount_orderings):
    '''
    Takes a graph and an amount_orderings. Returns a list of orderings based on recursive application of PLM.
    This list has the length amount_orderings.
    '''

    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        i = 0
        before = pd.Timestamp.now()

    root = contraction_trees.recursive_PLM(g)

    orderings = get_orderings(root, amount_orderings, connected_random_reorder_func(g))

    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        after = pd.Timestamp.now()
        print("Total time: {:f}s".format((after-before).total_seconds()))

    return orderings


def affinity_orderings(g, amount_orderings):
    '''
    Takes a graph and an amount_orderings. Returns a list of orderings based on the affinity contraction tree.
    This list has the length amount_orderings.
    '''

    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        i = 0
        before = pd.Timestamp.now()

    root = contraction_trees.affinity_tree(g)

    orderings = get_orderings(root, amount_orderings, connected_random_reorder_func(g))

    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        after = pd.Timestamp.now()
        print("Total time: {:f}s".format((after-before).total_seconds()))

    return orderings


def algebraic_distance_orderings(g, amount_orderings):
    '''
    Takes a graph and returns an ordering based on algebraic distances.
    '''
    # TODO implement correctly when algebraic distances are more promising
    raise NotImplementedError("Might be implemented when it looks more promising")

# Stuck at the end because of parse order...
ORD_ALG = dict(zip(config.ORD_TYPE, [recursive_PLM_orderings, affinity_orderings]))
