import networkit as nk
from fa2 import ForceAtlas2
import config


def map_to_line(points, coefficients):
    '''
    Maps points to a line. The amount of dimensions is implicitly given by the length of coefficients.
    Points is a list where [i*len(coefficients):(i+1)len(coefficients)] are the coordinates of the point i.
    '''

    res = list()
    for i in range(0, len(points), len(coefficients)):
        res.append(sum(p*c for p, c in zip(points[i:i+len(coefficients)], coefficients)))
    return res

def sort_nodes(weights):
    '''
    Sorts nodes according to their supplied weights.
    Input: List of weights. weights[n] == w means node n has weight w.
    Output: List of sorted nodes. ret[p] == n means node n has position p in the ordering.
    '''

    return [n for w, n in sorted(zip(weights, range(len(weights))))]

def algebraic_distance_orderings(g, coefficients_list):
    '''
    Returns orderings based on the algebraic distance. coefficients_list is a list of coefficients for the mapping.
    The length of coefficients_list is the amount of orderings returned.
    '''

    dist = nk.distance.AlgebraicDistance(g, numberSystems=config.ALG_DIST_SYSTEMS)
    dist.preprocess()
    ret = list()
    for coef in coefficients_list:
        ret.append(sort_nodes(map_to_line(dist.getLoads(), coef)))
    return ret

def force_atlas_2_orderings(g, coefficients_list):
    '''
    Returns orderings based on force atlas 2. coefficients_list is a list of coefficients for the mapping.
    Each entry must contain two coefficients. The length of coefficients_list is the amount of orderings returned.
    '''

    forceatlas2 = ForceAtlas2(
                        # Behavior alternatives
                        outboundAttractionDistribution=True,  # Dissuade hubs
                        linLogMode=False,  # NOT IMPLEMENTED
                        adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                        edgeWeightInfluence=1.0,

                        # Performance
                        jitterTolerance=1.0,  # Tolerance
                        barnesHutOptimize=True,
                        barnesHutTheta=1.2,
                        multiThreaded=False,  # NOT IMPLEMENTED

                        # Tuning
                        scalingRatio=2.0,
                        strongGravityMode=False,
                        gravity=1.0,

                        # Log
                        verbose=False)
    sparse_g = nk.algebraic.adjacencyMatrix(g)
    positions = forceatlas2.forceatlas2(sparse_g, pos=None, iterations=config.FORCEATLAS2_ITER)
    ret = list()
    for coef in coefficients_list:
        ret.append(sort_nodes(map_to_line([pos for coordinate in positions for pos in coordinate], coef)))
    return ret
