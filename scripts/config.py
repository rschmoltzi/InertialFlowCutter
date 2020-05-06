'''Configurations for the test scripts'''

from enum import IntEnum

class TimeStamps(IntEnum):
    NONE = 0
    SPARSE = 1
    SOME = 2
    ALL = 3


CONSOLE = "../cmake-build-release/console"

PATH = "../affinity/"

GRAPH_SUB = "walshaw/"
GRAPH_DIR = PATH + GRAPH_SUB
GRAPH_EXT = ".graph"

ORD_SUB = "orders/"
ORD_DIR = PATH + ORD_SUB
ORD_EXT = ".ord"

AMOUNT_ORDERS = 3
EPSILONS = [0.0, 0.01, 0.03, 0.05]
SEED = 31415
TIME_STAMPS = TimeStamps.SPARSE

ORD_TYPE = {"plm":"-PLM", "affinity":"-aff"}

DELIMITER_NODE = ","
DELIMITER_ORDER = "\n"
PLM_RESOLUTION = 0.01
