import subprocess, orderings, math, io, config, sys
import pandas as pd
from os import scandir, path, mkdir


def main():
    '''
    Computes a cch a evaluates the corresponding chordal super graph. The arguments are the graph to be tested and its ordering to be used.
    Times the computation of the cch.
    '''
    ordering_list = [["alg_dist", 10], ["asc_affinity", 10], ["plm", 10], ["asc_accumulated", 10]]

    config.ALG_DIST_ITER = 10000

    nested_dissection_experiments(ordering_list)

def nested_dissection_experiments(ordering_list):
    nd_dir = config.ORD_DIR + "nested-dissection/"
    if not path.isdir(nd_dir):
        mkdir(nd_dir)

    for ordering in ordering_list:
        config.AMOUNT_ORDERINGS = ordering[1]
        print()
        print()
        print("----------------")
        print(ordering[0])
        print()

        for entry in scandir(config.GRAPH_DIR):
            if entry.name.endswith(config.GRAPH_EXT):
                ord_start = pd.Timestamp.now()

                name = strip_ext(entry.name, config.GRAPH_EXT)
                ord_path = nd_dir + name + "_" + str(ordering[1]) + config.ORD_EXT
                connected = orderings.calculate_and_save_ordering(config.GRAPH_DIR + entry.name, ord_path, orderings.ORD_ALG[ordering[0]], config.AMOUNT_ORDERINGS)

                if not connected:
                    continue

                ord_end = pd.Timestamp.now()

                print("graph:", name)
                print("ord time:", (ord_end-ord_start).total_seconds())
                print()

                # Calctulate cch
                print(run(args_cch(name, ord_path)))



def strip_ext(name, ext):
    if name.endswith(ext):
        return name[:-(len(ext))]
    return name



def args_cch(graph_name, ord_path):
    args = [config.CONSOLE]

    args.append("load_metis_graph")
    args.append(config.get_graph_path(graph_name))

    args.append("load_node_orderings_cch")
    args.append(ord_path)

    args.append("add_back_arcs")
    args.append("remove_multi_arcs")
    args.append("remove_loops")

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

    args.append("report_time")
    args.append("reorder_nodes_in_accelerated_flow_cutter_cch_order_from_orderings")

    args.append("do_not_report_time")
    args.append("examine_chordal_supergraph")

    return args

def run(args):
    output = subprocess.check_output(args, universal_newlines=True)
    return output

if __name__ == '__main__':
    main()
