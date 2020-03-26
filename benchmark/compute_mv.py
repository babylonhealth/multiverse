
"""
Contains both `MultiVerse, not optimised`
and `MultiVerse, optimised` experiment code.
"""

import sys
sys.path.append("..")

import timeit

from config import MULTIVERSE_NUM_CORES_TO_USE
from multiverse.engine import (
    BernoulliERP,
    ObservableBernoulliERP,
    observe,
    do,
    predict,
    run_inference,
)
from experiment_utils import load_experiment, toposort
from utils import calculate_expectation


def create_prob_proc_object(model, var_name, dict_values):
    if model[var_name]['type'] == "exogenous":
        return BernoulliERP(
            prob=model[var_name]['prior'],
            depends_on=[],
            trace_address=var_name,
        )
    elif model[var_name]['type'] == "endogenous":
        weights = model[var_name]['parameters']
        parents = model[var_name]['parents']
        sum_parents = 0
        for index, parent in enumerate(parents):
            sum_parents += weights[index] * dict_values[parent].value

        if sum_parents > 0.5:  # activation function
            val = 1
        else:
            val = 0

        return ObservableBernoulliERP(
            input_val=val,
            noise_flip_prob=model[var_name]['flip_noise'],
            depends_on=[dict_values[parent] for parent in parents],
            noise_depends_on=[],
            trace_address=var_name,
        )
    else:
        raise ValueError


def program(model, toposorted_nodes, evidence, intervention, var_to_predict):
    dict_values = {}
    for node in toposorted_nodes:
        dict_values[node] = create_prob_proc_object(model, node, dict_values)
    for var_name, val in evidence.items():
        observe(dict_values[var_name], val)
    for var_name, val in intervention.items():
        do(dict_values[var_name], val)
    predict(dict_values[var_to_predict].value, predict_counterfactual=True)


def run_mv_non_opt(experiment_id, num_samples):
    model, intervention, evidence, var_to_predict = load_experiment(experiment_id)
    toposorted_nodes = toposort(model)

    start = timeit.default_timer()
    results = run_inference(
        program,
        num_samples,
        num_cores=MULTIVERSE_NUM_CORES_TO_USE,
        model_function_kwargs={
            'model': model,
            'toposorted_nodes': toposorted_nodes,
            'evidence': evidence,
            'intervention': intervention,
            'var_to_predict': var_to_predict,
        },
    )
    stop = timeit.default_timer()
    time_took = stop - start
    result = calculate_expectation(results)
    return result, time_took


###


# Now, code for `MultiVerse, optimised`:

from multiverse.engine import (
    IF_OBSERVE_BLOCK,
    IF_DO_BLOCK,
    compute_procedure_if_necessary,
)

def compute_var_helper(model, var_name, dict_values):
    for parent in model[var_name]['parents']:
        if parent not in dict_values:  # save some calls by check
            compute_var(model, dict_values, parent)  # compute parents

    return create_prob_proc_object(model, var_name, dict_values)


def compute_var(model, dict_values, var_name):
    if var_name in dict_values:
        return dict_values[var_name]

    dict_values[var_name] = compute_procedure_if_necessary(
        var_name,
        lambda: compute_var_helper(model, var_name, dict_values)
    )

    return dict_values[var_name]


def optimised_program(model, evidence, intervention, var_to_predict):
    dict_values = {}

    if IF_OBSERVE_BLOCK():
        for var_name, val in evidence.items():
            observe(
                compute_var(model, dict_values, var_name),
                val
            )

    if IF_DO_BLOCK():
        for var_name, val in intervention.items():
            do(
                compute_var(model, dict_values, var_name),
                val
            )

    predict(compute_var(model, dict_values, var_to_predict).value, predict_counterfactual=True)


def run_mv_opt(experiment_id, num_samples):
    model, intervention, evidence, var_to_predict = load_experiment(experiment_id)
    toposorted_nodes = toposort(model)

    start = timeit.default_timer()
    results = run_inference(
        optimised_program,
        num_samples,
        num_cores=MULTIVERSE_NUM_CORES_TO_USE,
        model_function_kwargs={
            'model': model,
            'evidence': evidence,
            'intervention': intervention,
            'var_to_predict': var_to_predict,
        },
    )
    stop = timeit.default_timer()
    time_took = stop - start
    result = calculate_expectation(results)
    return result, time_took
