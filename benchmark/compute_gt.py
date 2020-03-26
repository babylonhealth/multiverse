
"""
We compute the ground truth values of counterfactual queries
using enumeration.

To compute, we enumerate over all possible combinations
of exogenous variables twice:

1. First time we execute the program with each combination
   without interventions to calculate the posterior log-probabilities
   (incorporating priors and observation likelihoods).
   It is called `first_run`.

2. Second time we execute the program with each combination
   with interventions to get the predicted counterfactual values.
   It is called `second_run`.

Then, we weight the counterfactual predictions with log-probabilities
from the posterior. By doing so, we receive the counterfactual query
result.
"""

import itertools
import math
import numpy
import json
import os
from tqdm import tqdm

try:
    from scipy.misc import logsumexp
except ImportError:
    from scipy.special import logsumexp


from experiment_utils import load_experiment, toposort


EXPERIMENT_RANGE = range(1000)


OUTPUT_FOLDER = "output/calculated_gts/"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def calc_prob_val(model, var_name, dict_values, configuration_dict):
    if model[var_name]['type'] == "exogenous":
        val = configuration_dict[var_name]
        prob = model[var_name]['prior'] if val else 1.0 - model[var_name]['prior']

        return prob, val

    elif model[var_name]['type'] == "endogenous":
        weights = model[var_name]['parameters']
        parents = model[var_name]['parents']
        sum_parents = 0
        for index, parent in enumerate(parents):
            sum_parents += weights[index] * dict_values[parent]

        if sum_parents > 0.5:  # activation function
            val = 1
        else:
            val = 0

        # configuration_dict[var_name] is the value of its "noisy flipper"
        # parent (which is exogenous) rather than this endogenous variable
        # itself
        if configuration_dict[var_name] == 1:
            if val == 0:
                val = 1
            else:
                val = 0

        # this is not the probability of this variable but of the "noise flipper"
        if configuration_dict[var_name]:
            prob = model[var_name]['flip_noise']
        else:
            prob = 1.0 - model[var_name]['flip_noise']

        return prob, val

    else:
        raise ValueError


def program(
    mode, model, toposorted_nodes, configuration_dict,
    intervention, evidence, var_to_predict,
):
    log_prob = 0.0
    dict_values = {}
    for node in toposorted_nodes:
        if mode == "first_run" or node not in intervention:
            prob, dict_values[node] = calc_prob_val(
                model, node, dict_values, configuration_dict
            )
            log_prob += math.log(prob)
        else:
            dict_values[node] = intervention[node]

    for var_name, val in evidence.items():
        if dict_values[var_name] != val:
            # Rejecting this sample completely:
            log_prob = float("-inf")

    if mode == "first_run":
        return log_prob
    else:
        return dict_values[var_to_predict]


def gen_state_space(toposorted_nodes):
    return itertools.product((0, 1), repeat=len(toposorted_nodes))


def extract_state_space_configuration_as_dict(configuration, toposorted_nodes, index):
    configuration_dict = {}
    for node_index, node in enumerate(toposorted_nodes):
        configuration_dict[node] = configuration[node_index]

    return configuration_dict


for EXPERIMENT_ID in tqdm(EXPERIMENT_RANGE):
    model, intervention, evidence, var_to_predict = load_experiment(EXPERIMENT_ID)
    toposorted_nodes = toposort(model)


    logprobs = []

    for index, configuration in enumerate(gen_state_space(toposorted_nodes)):
        # The values of a particular state space configurations.

        configuration_dict = extract_state_space_configuration_as_dict(
            configuration, toposorted_nodes, index
        )

        # (Note: for endogenous variables, these are the values of
        #  its noisy-flipper rather than the endogenous node values themselves.)
        
        log_prob = program(
            "first_run", model, toposorted_nodes, configuration_dict,
            intervention, evidence, var_to_predict,
        )

        logprobs += [log_prob]


    values = []

    for index, configuration in enumerate(gen_state_space(toposorted_nodes)):
        if logprobs[index] != float("-inf"):
            configuration_dict = extract_state_space_configuration_as_dict(
                configuration, toposorted_nodes, index
            )

            value = program(
                "second_run", model, toposorted_nodes, configuration_dict,
                intervention, evidence, var_to_predict,
            )
            values += [value]
        else:
            values += [0]  # the value does not matter as the weight is 0.0


    assert any([el for el in logprobs if el > float("-inf")])

    values_by_weights = logsumexp(a=logprobs, b=values)
    weights = logsumexp(a=logprobs)
    expectation = numpy.exp(values_by_weights - weights)

    json.dump(expectation, open(OUTPUT_FOLDER + str(EXPERIMENT_ID) + ".json", "w"))

