
import argparse
import matplotlib.pyplot as plt
from networkx import DiGraph
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
import graphviz as pgv
import os

from experiment_utils import load_experiment


OUTPUT_FOLDER = "output/"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


parser = argparse.ArgumentParser(description='Visualise a network')
parser.add_argument(
    '--exp_id', required=False, type=int, default=0,
    help='an experiment id of a network',
)
args = parser.parse_args()

model, intervention, evidence, var_to_predict = load_experiment(args.exp_id)

Graph = DiGraph()

def get_noisy_flipper_name(node_id):
    return node_id + "_NF"

all_nodes = []
for node_id, node_data in model.items():
    all_nodes.append(node_id)
    if node_data['type'] == "endogenous":
        # Noisy Flippers:
        all_nodes.append(get_noisy_flipper_name(node_id))

Graph.add_nodes_from(model.keys())
for child_node_id, child_node_data in model.items():
    for parent_id in child_node_data['parents']:
        Graph.add_edge(parent_id, child_node_id)
    if child_node_data['type'] == "endogenous":
        Graph.add_edge(get_noisy_flipper_name(child_node_id), child_node_id)

pygraphviz_Graph = to_agraph(Graph)
pygraphviz_Graph.layout('dot')
pygraphviz_Graph.draw(OUTPUT_FOLDER + 'network.png')
