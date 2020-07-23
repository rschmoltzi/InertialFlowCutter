import networkit as nk
import contraction_trees, algebraic_distances, config, random
from anytree import Node, RenderTree, PreOrderIter, AnyNode
from anytree.iterators import AbstractIter
from collections import Counter, OrderedDict
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

def calculate_and_save_ordering(path_to_graph, path_to_ord, ordering_alg, amount_orderings, reader=nk.graphio.METISGraphReader()):
    '''
    Calculates the orderings for the given graph and saves them in the specified directory.
    '''
    g = load_graph(path_to_graph, reader)

    # Test to check for not connected graphs. Remove for more performance
    comp = nk.components.ConnectedComponents(g)
    comp.run()
    if comp.numberOfComponents() > 1:
        return False

    orders = ordering_alg(g, amount_orderings)

    with open(path_to_ord, "w") as f:
        f.write(config.DELIMITER_ORDER.join(config.DELIMITER_NODE.join(str(i) for i in order) for order in orders))

    return True


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
    ordering = [node.node_id for node in RandomDFSIter(root) if node.is_leaf]

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

def increment_edge_count(node):
    '''
    Increments the count attribute of the given Node and all its ancestors.
    '''
    current_node = node
    while True:
        current_node.count += 1
        if current_node.is_root:
            break
        current_node = current_node.parent


def rec_reorder_clusters_with_common_parent(g, parent, node_to_leaf):
    '''
    Reorders all children of the given node. It selects the child with the highest edge count to the already visited nodes.
    If the child is a leaf, its neighbor edges will be counted and added to the rest. Otherwise it will recursively call
    itself with the current node as new parent.
    '''

    clusters = list(parent.children)
    ret = list()
    while clusters:
        current_node = max(clusters, key=lambda node: node.count)
        if current_node.is_leaf:
            for n in g.neighbors(current_node.node_id):
                increment_edge_count(node_to_leaf[n])

            ret.append(current_node)
        else:
            ret += rec_reorder_clusters_with_common_parent(g, current_node, node_to_leaf)

        clusters.remove(current_node)

    return ret


def ascending_connected_random_orderings(g, root, n): # TODO: Increase performance
    '''
    Pseudo randomly reorders a list of clusters based on their connectivity in g.
    Starts with a random DFS until reaching the lowest level. Then adds clusters based on their connectivity.
    '''

    random.seed(config.SEED)

    node_to_leaf = OrderedDict() # map from node to leaf
    for leaf in root.leaves:
        node_to_leaf[leaf.node_id] = leaf

    # the count represents how many edges reach this cluster.
    # will be updated throughout the tree
    ret = list()

    for i in range(n):
        for node in PreOrderIter(root):
            node.count = 0

        current_leaf = node_to_leaf[peripheral_node(g, nk.graphtools.randomNode(g))]

        increment_edge_count(current_leaf)

        ret.append([node.node_id for node in rec_reorder_clusters_with_common_parent(g, root, node_to_leaf)])

    for node in PreOrderIter(root):
        del node.count

    return ret



def random_reorder(l):
    return random.sample(l, len(l))

def id(l):
    return l

# Returns a function that realizes a randomized-connected reordering on the given graph g
def connected_random_reorder_func(g):
    '''
    Wrapper function used to put g in the namespace of the returned function without changing its signature.
    Returns a function that pseudo randomly reorders a list of clusters based on their connectivity in g.
    '''

    def connected_random_reorder(l):
        if len(l) <= 1:
            return l

        nodes = OrderedDict()

        # inverse map
        node_to_cluster = OrderedDict()
        for cluster in l:
            cluster_to_nodes = set()
            for node in PreOrderIter(cluster):
                if node.is_leaf:
                    node_to_cluster[node.node_id] = cluster
                    cluster_to_nodes.add(node.node_id)

            nodes[cluster] = cluster_to_nodes

        #choose random cluster c
        c = random.choice(l)
        # used as set that preserves insertion order
        added_clusters = OrderedDict()
        added_clusters[c] = None

        #create list with #edges to the cluster
        edge_count = Counter()
        # Initialization is needed, since some of the plm clusters seem to be not connected
        for cluster in l:
            edge_count[cluster] = 0

        while True:
            del edge_count[c] # stops the code from adding the same cluster twice

            for u in nodes[c]:
                for n in g.neighbors(u):
                    if n in node_to_cluster and node_to_cluster[n] not in added_clusters:
                        edge_count[node_to_cluster[n]] += 1

            c = edge_count.most_common(1)[0][0] # list of tuples
            added_clusters[c] = None
            if len(added_clusters) >= len(l):
                break

        return list(added_clusters.keys())

    return connected_random_reorder


def peripheral_node(g, start_node):
    '''
    Computes a peripheral node by computing two chained BFS form the start_node, each time selecting the furthest node.
    '''
    def bfs_callback(node, count):
        peripheral_node = node

    peripheral_node = start_node
    nk.graph.Traversal.BFSfrom(g, start_node, bfs_callback)

    intermediate_node = peripheral_node
    nk.graph.Traversal.BFSfrom(g, intermediate_node, bfs_callback)

    return peripheral_node

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


def recursive_PLM_orderings(g, amount_orderings):
    '''
    Takes a graph and an amount_orderings. Returns a list of orderings based on recursive application of PLM.
    This list has the length amount_orderings.
    '''

    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        i = 0
        before = pd.Timestamp.now()

    nk.setSeed(config.SEED, False)
    root = contraction_trees.recursive_PLM(g)
    orderings = get_orderings(root, amount_orderings, connected_random_reorder_func(g))

    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        after = pd.Timestamp.now()
        print("Total time: {:f}s".format((after-before).total_seconds()))

    return orderings


def accumulated_contraction_orderings(g, amount_orderings):
    '''
    Takes a graph and an amount_orderings. Returns a list of orderings. One half are plm orderings, the other
    affinity orderings. This list has the length amount_orderings.
    '''

    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        i = 0
        before = pd.Timestamp.now()

    orderings = affinity_orderings(g, amount_orderings-(amount_orderings//2))
    orderings += recursive_PLM_orderings(g, amount_orderings//2)

    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        after = pd.Timestamp.now()
        print("Total time: {:f}s".format((after-before).total_seconds()))

    return orderings


def ascending_affinity_orderings(g, amount_orderings):
    '''
    Takes a graph and an amount_orderings. Returns a list of orderings based on the affinity contraction tree.
    This list has the length amount_orderings. Reorders the leaves with a connectivity-based pseudorandom DFS.
    '''

    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        i = 0
        before = pd.Timestamp.now()

    nk.setSeed(config.SEED, False)

    root = contraction_trees.affinity_tree(g)
    orderings = ascending_connected_random_orderings(g, root, amount_orderings)

    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        after = pd.Timestamp.now()
        print("Total time: {:f}s".format((after-before).total_seconds()))

    return orderings


def ascending_recursive_PLM_orderings(g, amount_orderings):
    '''
    Takes a graph and an amount_orderings. Returns a list of orderings based on recursive application of PLM.
    This list has the length amount_orderings. Reorders the leaves with a connectivity-based pseudorandom DFS.
    '''

    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        i = 0
        before = pd.Timestamp.now()

    nk.setSeed(config.SEED, False)
    root = contraction_trees.recursive_PLM(g)
    orderings = ascending_connected_random_orderings(g, root, amount_orderings)

    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        after = pd.Timestamp.now()
        print("Total time: {:f}s".format((after-before).total_seconds()))

    return orderings


def ascending_accumulated_contraction_orderings(g, amount_orderings):
    '''
    Takes a graph and an amount_orderings. Returns a list of orderings. One half are plm orderings, the other
    affinity orderings. This list has the length amount_orderings. Reorders the leaves with a connectivity-based pseudorandom DFS.
    '''

    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        i = 0
        before = pd.Timestamp.now()

    nk.setSeed(config.SEED, False)

    orderings = ascending_affinity_orderings(g, amount_orderings-(amount_orderings//2))
    orderings += ascending_recursive_PLM_orderings(g, amount_orderings//2)

    if config.TIME_STAMPS >= config.TimeStamps.ALL:
        after = pd.Timestamp.now()
        print("Total time: {:f}s".format((after-before).total_seconds()))

    return orderings




# -------------  Position based orderings -------------------

def generate_random_coefficients(amount, dim):
    '''
    Generates a two dimensional list with a length of amount and each sublist has the length dim.
    Each entry is a random coefficient in [-1,1].
    '''

    ret = list()
    for i in range(amount):
        coefficients = list()
        for d in range(dim):
            coefficients.append(random.uniform(-1,1))

        ret.append(coefficients)

    return ret

def algebraic_distance_orderings(g, amount_orderings):
    '''
    Takes a graph and an amount_orderings. Returns a list of orderings based on mappings to pseudo random lines of the algebraic distances.
    This list has the length amount_orderings.
    '''

    random.seed(config.SEED)
    coefficients = generate_random_coefficients(amount_orderings, config.ALG_DIST_SYSTEMS)
    return algebraic_distances.algebraic_distance_orderings(g, coefficients)


def force_atlas_2_orderings(g, amount_orderings):
    '''
    Takes a graph and an amount_orderings. Returns a list of orderings based on mappings to pseudo random lines of the force atlas 2 positions.
    This list has the length amount_orderings.
    '''

    random.seed(config.SEED)
    coefficients = generate_random_coefficients(amount_orderings, 2)
    return algebraic_distances.force_atlas_2_orderings(g, coefficients)

# Stuck at the end because of parse order...
ORD_ALG = OrderedDict(zip(config.ORD_TYPE, [affinity_orderings, recursive_PLM_orderings, algebraic_distance_orderings, force_atlas_2_orderings, accumulated_contraction_orderings, ascending_affinity_orderings, ascending_recursive_PLM_orderings, ascending_accumulated_contraction_orderings]))
