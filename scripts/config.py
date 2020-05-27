'''Configurations for the test scripts'''

from enum import IntEnum

class TimeStamps(IntEnum):
    NONE = 0
    SPARSE = 1
    SOME = 2
    ALL = 3

# Paths and file details
CONSOLE = "../cmake-build-release/console"

PATH = "../affinity/"
GRAPH_SUB = "walshaw/"
GRAPH_DIR = PATH + GRAPH_SUB
GRAPH_EXT = ".graph"
ORD_SUB = "orders/"
ORD_DIR = PATH + ORD_SUB
ORD_EXT = ".ord"
ORD_TYPE = {"plm":"-PLM", "affinity":"-aff", "alg_dist":"-alg_dist", "fa2":"-fa2", "accumulated":"-acc"}

DELIMITER_NODE = ","
DELIMITER_ORDER = "\n"


# Test parameters
AMOUNT_ORDERS = 6
SEED = 31415
INITIAL_ASSIM = 0.05 # default: 0.05
BULK_STEP = 0.05 # default: 0.05

# Specific ordering parameters
PLM_RESOLUTION = 0.01 # Threshold for the recursion depth of the PLM ordering
FORCEATLAS2_ITER = 20 # Reasonably fast, but the results are bad
ALG_DIST_SYSTEMS = 10

# Output parameters
TIME_STAMPS = TimeStamps.SPARSE

# Evaluation parameters
EPSILONS = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
