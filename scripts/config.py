'''Configurations for the test scripts'''

from enum import IntEnum
import os
from collections import OrderedDict

class TimeStamps(IntEnum):
    NONE = 0
    SPARSE = 1
    SOME = 2
    ALL = 3

# Paths and file details
ABS_PATH_WD = os.getcwd() + "/"

CONSOLE = ABS_PATH_WD + "../cmake-build-release/console"

PATH = ABS_PATH_WD + "../graphs/"
GRAPH_SUB = "walshaw/"
GRAPH_DIR = PATH + GRAPH_SUB
GRAPH_EXT = ".graph"
ORD_SUB = "orderings/"
ORD_DIR = PATH + ORD_SUB
ORD_EXT = ".ord"
ORD_TYPE_DELIMITER = "-"
ORDERING_ALG_NAMES = ["affinity", "plm", "alg_dist", "fa2", "accumulated", "asc_affinity", "asc_plm", "asc_accumulated"] # must not contain the ORD_TYPE_DELIMITER
ORD_TYPE = OrderedDict(zip(ORDERING_ALG_NAMES, [ORD_TYPE_DELIMITER + alg for alg in ORDERING_ALG_NAMES]))

CSV_EVALUATION_DIR = "csv-data/"


DELIMITER_NODE = ","
DELIMITER_ORDER = "\n"


# Test parameters
AMOUNT_ORDERINGS = 3 # Default, when no command line argument is given
SEED = 31415 # Random seed for python and networkit
INITIAL_ASSIM = 0.05 # default: 0.05
BULK_STEP = 0.05 # default: 0.05

# Specific ordering parameters
PLM_RESOLUTION = 0.0001 # Threshold for the recursion depth of the PLM ordering
FORCEATLAS2_ITER = 20 # Reasonably fast, but the results are bad
ALG_DIST_SYSTEMS = 10
ALG_DIST_ITER = 100000

# Output parameters
TIME_STAMPS = TimeStamps.SPARSE

# Evaluation parameters
EPSILONS = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

def get_graph_path(name):
    return GRAPH_DIR + name + GRAPH_EXT

def get_ord_path(name, ord_rep):
    return ORD_DIR + name + ORD_TYPE[ord_rep] + ORD_EXT
