
from multiverse.engine import (
    BernoulliERP,
    DeltaERP,
    ObservableBernoulliERP,
    do,
    observe,
    predict,
    run_inference,
)
from utils import calculate_expectation


def flip_value(val):
    assert val in [0, 1]
    if val == 1:
        return 0
    else:
        return 1


def model_discrete__query_1():
    x = BernoulliERP(prob=0.0001, proposal_prob=0.1)
    z = BernoulliERP(prob=0.001, proposal_prob=0.1)
    noise_flipper = BernoulliERP(prob=0.00001, proposal_prob=0.1)
    x_or_z = x.value or z.value
    y = DeltaERP(flip_value(x_or_z) if noise_flipper.value else x_or_z)
    observe(y, 1)
    predict(x.value)


results = run_inference(model_discrete__query_1, 10000)
print("")
print("Posterior (observational) inference:")
print("ExpectedValue(X | Y = 1)")
print(calculate_expectation(results))
print("***")


# ###


def model_discrete__query_2():
    x = BernoulliERP(prob=0.0001, proposal_prob=0.1)
    z = BernoulliERP(prob=0.001, proposal_prob=0.1)
    y = ObservableBernoulliERP(
        input_val=x.value or z.value,
        noise_flip_prob=0.00001,
        depends_on=[x, z]
    )
    observe(y, 1)
    do(x, 0)
    predict(y.value)


results = run_inference(model_discrete__query_2, 10000)
print("")
print("Counterfactual inference:")
print("ExpectedValue(Y' | Y = 1, do(X = 1))")
print(calculate_expectation(results))
print("***")
