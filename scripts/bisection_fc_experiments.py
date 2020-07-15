import subprocess, orderings, math, io, config, sys
import pandas as pd
from os import scandir, path, mkdir
from config import get_graph_path, get_ord_path
from collections import OrderedDict



def main():
    '''
    Makes cut experiments and saves them as csv.
    '''

    fc_path = config.CSV_EVALUATION_DIR + "flow-cutter/"
    if not path.isdir(fc_path):
        mkdir(fc_path)

    df = summarize_data(enum_cuts_all_fc("asc_affinity")) # The param is used to avoid caluculating cuts which we didnt eval with ifc
    df.to_csv(path_or_buf=fc_path + "fc.csv")


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


    return summary


def enum_cuts_all_fc(ord_rep):
    '''
    Enumerates the cuts for all graph orderings in config.GRAPH_DIR, if they have a representative in config.ORD_DIR.
    '''
    if config.TIME_STAMPS >= config.TimeStamps.SPARSE:
        before = pd.Timestamp.now()

    data = OrderedDict()

    for entry in scandir(config.GRAPH_DIR):
        name = strip_ext(entry.name,config.GRAPH_EXT)
        if path.isfile(config.get_ord_path(name, ord_rep)):
            if config.TIME_STAMPS >= config.TimeStamps.SOME:
                entry_start = pd.Timestamp.now()

            data[name] = run(args_enum_cuts_fc(name))

            if config.TIME_STAMPS >= config.TimeStamps.SOME:
                entry_end = pd.Timestamp.now()
                print("Calculating cut on " + entry.name + ": {:f}s".format((entry_end-entry_start).total_seconds()))

    if config.TIME_STAMPS >= config.TimeStamps.SPARSE:
        after = pd.Timestamp.now()
        print("Calculating cuts: {:f}s".format((after-before).total_seconds()))

    return data



def args_enum_cuts_fc(name):
    args = [config.CONSOLE]

    args.append("load_metis_graph")
    args.append(get_graph_path(name))

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
    args.append("cutter_count")
    args.append(str(20))

    args.append("flow_cutter_enum_cuts")
    args.append("-")

    return args


def run(args):
    output = subprocess.check_output(args, universal_newlines=True)
    rename = {'    time' : 'time'}
    ret = pd.read_csv(io.StringIO(output)).rename(rename, axis='columns')

    return ret


if __name__ == '__main__':
    main()
