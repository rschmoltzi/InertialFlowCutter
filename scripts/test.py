import subprocess, affinity_order, math,io
import pandas as pd
from os import scandir

console = "../cmake-build-release/console"

PATH = "../affinity/"

GRAPH_SUB = "walshaw/"
GRAPH_DIR = PATH + GRAPH_SUB
GRAPH_EXT = ".graph"

ORD_SUB = "orders/"
ORD_DIR = PATH + ORD_SUB
ORD_TYPE = "-aff"
ORD_EXT = ".ord"

EPSILONS = [0.0, 0.01, 0.03, 0.05]

def main():
    # calculate_all_orders()
    # print("\n\n".join(enum_cuts_all()))

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(enum_cuts_all())


def get_graph_path(name):
    return GRAPH_DIR + name + GRAPH_EXT

def get_ord_path(name):
    return PATH + ORD_SUB + name + ORD_TYPE + ORD_EXT

def strip_ext(name, ext):
    if name.endswith(ext):
        return name[:-(len(ext))]
    return name

def get_eps_label(eps):
    return "Eps = {0:.0%}".format(eps)

def calculate_eps(small_side, large_side):
    if (abs(small_side - large_side) <= 1):
        return 0.0
    else:
        return 2 * (large_side / (large_side + small_side)) - 1

def fit_row_in_summary_epsilons(summary_row, row):
    eps = calculate_eps(row["small_side_size"], row["large_side_size"])
    for cmp_eps in EPSILONS:
        if eps <= cmp_eps:
            if math.isnan(summary_row[get_eps_label(cmp_eps)]) or row["cut_size"] < summary_row[get_eps_label(cmp_eps)]:
                summary_row[get_eps_label(cmp_eps)] = row["cut_size"]

def summarize_data(data):
    col = list(map(get_eps_label, EPSILONS))
    col.append("Time IFC")
    summary = pd.DataFrame(index=data, columns=col)
    for name, frame in data.items():
        summary.loc[name, "Time IFC"] = frame["time"].max()
        for ind, row in frame.iterrows():
            fit_row_in_summary_epsilons(summary.loc[name, :], row)

    return summary

def enum_cuts_all():
    before = pd.Timestamp.now()

    data = dict()

    for entry in scandir(ORD_DIR):
        if entry.name.endswith(ORD_EXT):
            entry_start = pd.Timestamp.now()

            name = strip_ext(entry.name, ORD_TYPE + ORD_EXT)
            data[name] = run(args_enum_cuts(name))

            entry_end = pd.Timestamp.now()
            print("Calculating cut on " + entry.name + ": {:f}s".format((entry_end-entry_start).total_seconds()))

    after = pd.Timestamp.now()
    print("Calculating cuts: {:f}s".format((after-before).total_seconds()))

    return summarize_data(data)



def calculate_all_orders():
    before = pd.Timestamp.now()

    for entry in scandir(GRAPH_DIR):
        if entry.name.endswith(GRAPH_EXT):
            entry_start = pd.Timestamp.now()
            print("Starting order calculation for " + entry.name)

            name = strip_ext(entry.name, GRAPH_EXT)
            affinity_order.calculate_and_save_order(get_graph_path(name), get_ord_path(name))

            entry_end = pd.Timestamp.now()
            print("Calculating order for " + entry.name + ": {:f}s".format((entry_end-entry_start).total_seconds()))

    after = pd.Timestamp.now()
    print("Calculating orders: {:f}s".format((after-before).total_seconds()))

def args_enum_cuts(name):
    args = [console]

    args.append("load_metis_graph")
    args.append(get_graph_path(name))

    args.append("load_node_orders")
    args.append(get_ord_path(name))

    args.append("add_back_arcs")
    args.append("remove_multi_arcs")
    args.append("remove_loops")
    args.append("reorder_nodes_in_preorder")
    args.append("sort_arcs")

    args.append("flow_cutter_set")
    args.append("random_seed")
    args.append("5489")

    args.append("flow_cutter_set")
    args.append("ReportCuts")
    args.append("no")

    args.append("flow_cutter_set")
    args.append("max_cut_size")
    args.append("1000000") # default:1000 m14b:4000

    args.append("flow_cutter_accelerated_enum_cuts_from_orders")
    args.append("-")

    return args

def run(args):
    output = subprocess.check_output(args, universal_newlines=True)
    rename = {'    time' : 'time'}
    return pd.read_csv(io.StringIO(output)).rename(rename, axis='columns')

if __name__ == '__main__':
    main()
