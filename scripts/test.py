import subprocess

console = "../cmake-build-debug/console"

path = "../affinity/"
graph_name = "test_path"
graph_ext = ".graph"
graph_path = path + graph_name + graph_ext

ord_type = "-aff"
ord_ext = ".ord"
ord_path = path + graph_name + ord_type + ord_ext

def test_ifc():
    args = [console]

    args.append("load_metis_graph")
    args.append(graph_path)

    args.append("load_node_orders")
    args.append(ord_path)

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
    args.append("4000") # default:1000 m14b:4000

    args.append("flow_cutter_accelerated_enum_cuts_from_orders")
    args.append("-")

    run(args)

def run(args):
    output = subprocess.check_output(args, universal_newlines=True)
    print(output)
    # if output.returncode == 0:
    #     print(output.output)
    # else:
    #     print("Error, console crashed")

test_ifc()
