import subprocess, orderings, math, io, config, sys
import pandas as pd
from os import scandir


def main():
    '''
    Starts the IFC tests for one specified ordering algorithm. The graphs that will be cut must be in the directory
    specified in config.py and must be in the METIS format.
    '''

    if len(sys.argv) != 2:
        raise AttributeError("You must specify exactly one ordering algorithm.")
    if sys.argv[1] not in config.ORD_TYPE:
        raise ValueError("The given argument does not represent an ordering algorithm")

    ord_rep = sys.argv[1]

    calculate_all_orders(orderings.ORD_ALG[sys.argv[1]], ord_rep)


    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    ret = enum_cuts_all(ord_rep)

    #Can be used to create a pretty HTML output
    #ret.style.format({'Time IFC': "{:.2f}"})
    print(ret)


def get_graph_path(name):
    return config.GRAPH_DIR + name + config.GRAPH_EXT

def get_ord_path(name, ord_rep):
    return config.PATH + config.ORD_SUB + name + config.ORD_TYPE[ord_rep] + config.ORD_EXT

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
    for cmp_eps in config.EPSILONS:
        if eps <= cmp_eps:
            if math.isnan(summary_row[get_eps_label(cmp_eps)]) or row["cut_size"] < summary_row[get_eps_label(cmp_eps)]:
                summary_row[get_eps_label(cmp_eps)] = row["cut_size"]

def summarize_data(data):
    col = list(map(get_eps_label, config.EPSILONS))
    col.append("Time IFC")
    summary = pd.DataFrame(index=sorted(data), columns=col)
    for name, frame in data.items():
        summary.loc[name, "Time IFC"] = frame["time"].max() / 1000000
        for ind, row in frame.iterrows():
            fit_row_in_summary_epsilons(summary.loc[name, :], row)

    return summary

def enum_cuts_all(ord_rep):
    if config.TIME_STAMPS >= config.TimeStamps.SPARSE:
        before = pd.Timestamp.now()

    data = dict()

    for entry in scandir(config.ORD_DIR):
        if entry.name.endswith(config.ORD_TYPE[ord_rep] + config.ORD_EXT):
            if config.TIME_STAMPS >= config.TimeStamps.SOME:
                entry_start = pd.Timestamp.now()

            name = strip_ext(entry.name, config.ORD_TYPE[ord_rep] + config.ORD_EXT)
            data[name] = run(args_enum_cuts(name, ord_rep))

            if config.TIME_STAMPS >= config.TimeStamps.SOME:
                entry_end = pd.Timestamp.now()
                print("Calculating cut on " + entry.name + ": {:f}s".format((entry_end-entry_start).total_seconds()))

    if config.TIME_STAMPS >= config.TimeStamps.SPARSE:
        after = pd.Timestamp.now()
        print("Calculating cuts: {:f}s".format((after-before).total_seconds()))

    return summarize_data(data)



def calculate_all_orders(ordering_alg, ord_rep):
    if config.TIME_STAMPS >= config.TimeStamps.SPARSE:
        before = pd.Timestamp.now()

    for entry in scandir(config.GRAPH_DIR):
        if entry.name.endswith(config.GRAPH_EXT):
            if config.TIME_STAMPS >= config.TimeStamps.SOME:
                entry_start = pd.Timestamp.now()

            name = strip_ext(entry.name, config.GRAPH_EXT)
            orderings.calculate_and_save_order(get_graph_path(name), get_ord_path(name, ord_rep), ordering_alg, config.AMOUNT_ORDERS)

            if config.TIME_STAMPS >= config.TimeStamps.SOME:
                entry_end = pd.Timestamp.now()
                print("Calculating order for " + entry.name + ": {:f}s".format((entry_end-entry_start).total_seconds()))

    if config.TIME_STAMPS >= config.TimeStamps.SPARSE:
        after = pd.Timestamp.now()
        print("Calculating orders: {:f}s".format((after-before).total_seconds()))

def args_enum_cuts(name, ord_rep):
    args = [config.CONSOLE]

    args.append("load_metis_graph")
    args.append(get_graph_path(name))

    args.append("load_node_orders")
    args.append(get_ord_path(name, ord_rep))

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
