
import itertools
import copy
import math
import numpy

try:
    from scipy.misc import logsumexp
except ImportError:
    from scipy.special import logsumexp


random_variables = [
    "RiskFactor",
    "Disease_One",
    "Disease_Two",
    "Disease_Three",
    "Symptom_Leak",
    "Disease_One_On_CausingSymptom",
    "Disease_Two_On_CausingSymptom",
]


endogenous_variables = [
    "Symptom"
]


prior_cpts = [
    lambda state: 0.01,
    lambda state: 0.2 if state["RiskFactor"] else 0.000001,
    lambda state: 0.2 if state["RiskFactor"] else 0.000001,
    lambda state: 0.95 if state["RiskFactor"] else 0.0001,
    lambda state: 0.0001,
    lambda state: 0.5,
    lambda state: 0.5,
]


def calc_symptom_val(configuration):
    # Noisy OR:

    symptom_val = 0
    
    if (
        configuration["Disease_One"] and
        configuration["Disease_One_On_CausingSymptom"]
    ):
        symptom_val = 1
    
    if (
        configuration["Disease_Two"] and
        configuration["Disease_Two_On_CausingSymptom"]
    ):
        symptom_val = 1

    if configuration['Symptom_Leak']:
        symptom_val = 1

    return symptom_val


endogenous_functions = [
    calc_symptom_val
]


def gen_state_space(random_variables):
    return [el for el in itertools.product((0, 1), repeat=len(random_variables))]


def extract_state_space_configuration_as_dict(configuration_list):
    configuration_dict = {}
    for node_index, node in enumerate(random_variables):
        configuration_dict[node] = configuration_list[node_index]

    return configuration_dict


state_space = gen_state_space(random_variables)


def calc_expectation(
    logprobs, values,
):
    assert any([el for el in logprobs if el > float("-inf")])

    values_by_weights = logsumexp(a=logprobs, b=values)
    weights = logsumexp(a=logprobs)
    expectation = numpy.exp(values_by_weights - weights)

    return expectation


def calc_query(
    evidence,
    interventions,
    predict_counterfactual,
    var_to_predict,
):
    logprobs = []
    predictions = []

    for configuration_list in state_space:
        configuration = extract_state_space_configuration_as_dict(configuration_list)

        regular_sample_configuration = copy.deepcopy(configuration)

        twin_network_configuration = copy.deepcopy(configuration)

        for intervention_key, intervention_val in interventions.items():
            assert intervention_key in twin_network_configuration
            twin_network_configuration[intervention_key] = intervention_val

        for endogenous_name, endogenous_function in zip(
            endogenous_variables, endogenous_functions,
        ):
            regular_sample_configuration[endogenous_name] = endogenous_function(
                regular_sample_configuration
            )
            twin_network_configuration[endogenous_name] = endogenous_function(
                twin_network_configuration
            )

        logprob = 0.0

        for variable, prior_cpt in zip(random_variables, prior_cpts):
            prior = prior_cpt(regular_sample_configuration)

            if regular_sample_configuration[variable]:
                prob = prior
            else:
                prob = 1.0 - prior

            logprob += math.log(prob)

        for evidence_variable, evidence_val in evidence.items():
            assert evidence_val in [0, 1]
            if evidence_val != regular_sample_configuration[evidence_variable]:
                logprob = float("-inf")

        logprobs.append(logprob)

        if predict_counterfactual:
            predictions.append(twin_network_configuration[var_to_predict])
        else:
            predictions.append(regular_sample_configuration[var_to_predict])

    return calc_expectation(logprobs, predictions)

###

for disease_index in ["One", "Two", "Three"]:
    print(
        "P(Disease_{} = T | Symptom = T) = ".format(disease_index),
        calc_query(
            evidence={"Symptom": 1},
            interventions={},
            predict_counterfactual=False,
            var_to_predict="Disease_{}".format(disease_index),
        )
    )

###

for disease_index in ["One", "Two", "Three"]:
    print(
        "P(Symptom' = F | Symptom = T, do(Disease_{} = F)) = ".format(disease_index),
        1.0 - calc_query(
            evidence={"Symptom": 1},
            interventions={"Disease_{}".format(disease_index): 0},
            predict_counterfactual=True,
            var_to_predict="Symptom",
        )
    )

