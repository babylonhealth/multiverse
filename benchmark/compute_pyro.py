
import timeit
import pyro
import torch
import networkx as nx
import pyro.distributions as dist
import numpy
import math

from experiment_utils import load_experiment, toposort


class AllSamplesRejectedException(Exception):
    pass


def get_noisy_flipper_name(endogenous_var_name):
    return endogenous_var_name + "_exog_noise"


def run_pyro(experiment_id, num_samples, type_to_run):
    # Note that for Pyro we use this number of samples twice because we sample twice:
    # once for abduction step and another time for prediction step.
    # See the paper for more details.

    assert type_to_run == "pyro_with_guide" or type_to_run == "pyro_without_guide"

    model, intervention, evidence, var_to_predict = load_experiment(experiment_id)
    toposorted_nodes = toposort(model)

    USE_GUIDE = type_to_run == "pyro_with_guide"

    exog_sites = set()
    for var_name in toposorted_nodes:
        if model[var_name]['type'] == "exogenous":
            exog_sites.add(var_name)
        else:
            # a noisy flipper of an endogenous variable:
            exog_sites.add(get_noisy_flipper_name(var_name))
    exog_sites_list = list(exog_sites)


    def create_prob_proc_object__pyro(var_name, dict_values, data, guide, exog_values=None):
        if model[var_name]['type'] == "exogenous":
            if exog_values is not None and var_name in exog_values:
                dict_values[var_name] = exog_values[var_name]
            else:
                dict_values[var_name] = pyro.sample(
                    var_name, dist.Bernoulli(model[var_name]['prior']),
                    obs=data.get(var_name, None)
                )
        elif model[var_name]['type'] == "endogenous":
            weights = model[var_name]['parameters']
            parents = model[var_name]['parents']
            sum_parents = 0
            for index, parent in enumerate(parents):
                sum_parents += weights[index] * float(dict_values[parent])

            if sum_parents > 0.5:  # activation function
                val = 1
            else:
                val = 0

            exog_noise_parent_name = get_noisy_flipper_name(var_name)
            if exog_values is not None and exog_noise_parent_name in exog_values:
                dict_values[exog_noise_parent_name] = exog_values[exog_noise_parent_name]
            else:
                if USE_GUIDE and guide == True and var_name in data:
                    if round(float(val)) == round(float(data[var_name])):
                        noise_flipper_prob = 0.0
                    else:
                        noise_flipper_prob = 1.0
                else:
                    noise_flipper_prob = model[var_name]['flip_noise']

                dict_values[exog_noise_parent_name] = pyro.sample(
                    exog_noise_parent_name,
                    dist.Bernoulli(noise_flipper_prob),
                )

            if round(float(dict_values[exog_noise_parent_name])) == 1:
                val = 1 if val == 0 else 0

            dict_values[var_name] = pyro.sample(
                var_name,
                dist.Delta(torch.tensor(val).float()),
                obs=data.get(var_name, None)
            )


    def generative_model(data={}, exog_values=None):
        dict_values = {}
        for node in toposorted_nodes:
            create_prob_proc_object__pyro(node, dict_values, data, guide=False, exog_values=exog_values)

        return dict_values


    def guide(data={}):
        dict_values = {}
        for node in toposorted_nodes:
            create_prob_proc_object__pyro(node, dict_values, data, guide=True, exog_values=None)

        return dict_values


    def abduction(evidence, n_samples):
        evidence = {d: torch.tensor(evidence[d]).float() for d in evidence}
        guide_to_use = None
        if USE_GUIDE:
            guide_to_use = guide

        posterior = pyro.infer.Importance(generative_model, guide_to_use, n_samples)
        posterior.run(evidence)
        if math.isnan(float(posterior.get_ESS())):
            raise AllSamplesRejectedException
        posterior = pyro.infer.EmpiricalMarginal(posterior, sites=exog_sites_list)

        return posterior


    def intervention_prediction(node_of_interest, intervention, posterior, n_samples):
        intervention = {k: torch.tensor(intervention[k]).float().flatten() for k in intervention}
        intervened_model = pyro.do(generative_model, data=intervention)

        estimate = []
        for _ in range(n_samples):
            exog_values_ = posterior.sample()
            exog_values = {}
            for index, var_name in enumerate(exog_sites_list):
                if var_name not in intervention.keys():
                    exog_values[var_name] = exog_values_[index]

            intervened_model_with_values = intervened_model(exog_values=exog_values)
            result = intervened_model_with_values[node_of_interest]
            estimate.append(result)

        return estimate


    start = timeit.default_timer()
    posterior = abduction(evidence, num_samples)
    results = intervention_prediction(var_to_predict, intervention, posterior, num_samples)
    stop = timeit.default_timer()
    time_took = stop - start
    result = float(numpy.mean(results))
    return result, time_took
