
import json


def load_experiment(experiment_id):
    experiment_data = json.load(open("experiment_inputs/experiment_" + str(experiment_id) + ".json"))

    model = experiment_data['model']
    intervention = experiment_data['intervention']
    evidence = experiment_data['evidence']
    var_to_predict = experiment_data['variable_to_predict']

    return model, intervention, evidence, var_to_predict


def toposort(model):
    # Model: dict(node -> {'parents': [X, Y, Z, ...], ...})

    toposorted_nodes = []

    node_to_children = dict()
    node_to_parents = dict()
    nodes_without_parents = set()

    for node_id, node_data in model.items():
        node_to_parents[node_id] = set(node_data['parents'])

        for parent in node_data['parents']:
            if parent not in node_to_children:
                node_to_children[parent] = set()
            node_to_children[parent].add(node_id)

        if len(node_data['parents']) == 0:
            nodes_without_parents.add(node_id)

    while len(nodes_without_parents) > 0:
        node = nodes_without_parents.pop()

        toposorted_nodes.append(node)

        for child_node in node_to_children.get(node, []):
            node_to_parents[child_node].remove(node)
            if len(node_to_parents[child_node]) == 0:
                nodes_without_parents.add(child_node)

        del node_to_parents[node]
        node_to_children.pop(node, None)

    return toposorted_nodes
