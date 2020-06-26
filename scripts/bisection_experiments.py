import subprocess, orderings, math, io, config, sys
import pandas as pd
from os import scandir, path, mkdir
from config import get_graph_path, get_ord_path
from collections import OrderedDict


def main():
    '''
    Starts the IFC enum_cuts tests for one specified ordering algorithm. The graphs that will be cut must be in the directory
    specified in config.py and must be in the METIS format.

    Run it like this: python3 -B bisection_experiments.py ordering_alg [amount_orderings]
    '''

    if len(sys.argv) == 3 and sys.argv[2].isdigit():
        config.AMOUNT_ORDERINGS = int(sys.argv[2])
    elif len(sys.argv) != 2:
        raise AttributeError("You must specify exactly one ordering algorithm.")
    if sys.argv[1] not in config.ORD_TYPE:
        raise ValueError("The given argument does not represent an ordering algorithm")

    ord_rep = sys.argv[1]



    #Can be used to create a pretty HTML output
    #ret.style.format({'Time IFC': "{:.2f}"})
    print(cut_experiments(ord_rep))


def cut_experiments_all_ordering_algs(amount_orderings=6):
    '''
    Makes cut experiments and saves them as csv.
    '''
    if not path.isdir(config.CSV_EVALUATION_DIR):
        mkdir(config.CSV_EVALUATION_DIR)
    config.AMOUNT_ORDERINGS = amount_orderings
    for ord_rep in orderings.ORD_ALG:
        df = cut_experiments(ord_rep)
        # with open(config.CSV_EVALUATION_DIR + ord_rep, "w"):
        df.to_csv(path_or_buf=config.CSV_EVALUATION_DIR + ord_rep + "_" + str(amount_orderings))


def experiments_orderings_amount():
    for amount in [3,6,10,20]:
        cut_experiments_all_ordering_algs(amount)


def cut_experiments(ord_rep):
    '''
    First computes all orderings and saves them, then computes the cuts on the orderings.
    '''

    time_orderings = calculate_all_orders(orderings.ORD_ALG[ord_rep], ord_rep)
    data = enum_cuts_all(ord_rep)

    # I dont even know if this is needed for print output.
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    return summarize_data(data, time_orderings)


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

def summarize_data(data, time_orderings):
    '''
    Creates a pd.DataFrame with the graphs as rows and the epsilons and the time as columns.
    '''
    col = list(map(get_eps_label, config.EPSILONS))
    col.append("Time IFC")
    summary = pd.DataFrame(index=sorted(data), columns=col)
    for name, frame in data.items():
        summary.loc[name, "Time IFC"] = frame["time"].max() / 1000000
        for ind, row in frame.iterrows():
            fit_row_in_summary_epsilons(summary.loc[name, :], row)

    summary["Time Ord"] = [time_orderings[x] for x in sorted(time_orderings)]
    summary["Time Sum"] = summary[["Time IFC", "Time Ord"]].sum(axis=1)

    return summary



def enum_cuts_all(ord_rep):
    '''
    Enumerates the cuts for all graph orderings in config.ORD_DIR. Fails if there is no graph with the same name in config.GRAPH_DIR.
    '''
    if config.TIME_STAMPS >= config.TimeStamps.SPARSE:
        before = pd.Timestamp.now()

    data = OrderedDict()

    for entry in scandir(config.GRAPH_DIR):
        name = strip_ext(entry.name,config.GRAPH_EXT)
        if path.isfile(config.get_ord_path(name, ord_rep)):
            if config.TIME_STAMPS >= config.TimeStamps.SOME:
                entry_start = pd.Timestamp.now()

            data[name] = run(args_enum_cuts(name, ord_rep))

            if config.TIME_STAMPS >= config.TimeStamps.SOME:
                entry_end = pd.Timestamp.now()
                print("Calculating cut on " + entry.name + ": {:f}s".format((entry_end-entry_start).total_seconds()))

    if config.TIME_STAMPS >= config.TimeStamps.SPARSE:
        after = pd.Timestamp.now()
        print("Calculating cuts: {:f}s".format((after-before).total_seconds()))

    return data



def calculate_all_orders(ordering_alg, ord_rep):
    if config.TIME_STAMPS >= config.TimeStamps.SPARSE:
        before = pd.Timestamp.now()

    times = OrderedDict()

    for entry in scandir(config.GRAPH_DIR):
        if entry.name.endswith(config.GRAPH_EXT):
            entry_start = pd.Timestamp.now()

            name = strip_ext(entry.name, config.GRAPH_EXT)
            orderings.calculate_and_save_order(get_graph_path(name), get_ord_path(name, ord_rep), ordering_alg, config.AMOUNT_ORDERINGS)

            entry_end = pd.Timestamp.now()
            times[name] = (entry_end-entry_start).total_seconds()

            if config.TIME_STAMPS >= config.TimeStamps.SOME:
                print("Calculating order for " + entry.name + ": {:f}s".format((entry_end-entry_start).total_seconds()))

    if config.TIME_STAMPS >= config.TimeStamps.SPARSE:
        after = pd.Timestamp.now()
        print("Calculating orders: {:f}s".format((after-before).total_seconds()))

    return times

#
# def time_function(function, verbosity, output_string):
#     def timed_function():
#         if config.TIME_STAMPS >= verbosity:
#             before = pd.Timestamp.now()
#
#
#         if config.TIME_STAMPS >= verbosity:
#             after = pd.Timestamp.now()
#             print("Calculating orders: {:f}s".format((after-before).total_seconds()))


def args_enum_cuts(name, ord_rep):
    args = [config.CONSOLE]

    args.append("load_metis_graph")
    args.append(get_graph_path(name))

    args.append("load_node_orderings")
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
    args.append("1000000") # default is 1000 but wont produce cuts on denser graphs

    args.append("flow_cutter_set")
    args.append("initial_assimilated_fraction")
    args.append(str(config.INITIAL_ASSIM)) # default 0.05

    args.append("flow_cutter_set")
    args.append("bulk_step_fraction")
    args.append(str(config.BULK_STEP))

    args.append("flow_cutter_accelerated_enum_cuts_from_orderings")
    args.append("-")

    return args

def run(args):
    output = subprocess.check_output(args, universal_newlines=True)
    rename = {'    time' : 'time'}
    return pd.read_csv(io.StringIO(output)).rename(rename, axis='columns')

if __name__ == '__main__':
    main()
