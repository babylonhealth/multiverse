
from multiverse.engine import (
    NormalERP,
    ObservableNormalERP,
    do,
    observe,
    predict,
    run_inference,
)
from utils import calculate_expectation


def model_continuous__query_1():
    x = NormalERP(mean=0, std=1)
    z = NormalERP(mean=0, std=1)
    y = NormalERP(mean=x.value + z.value, std=2, depends_on=[x, z])
    observe(y, 3)
    predict(x.value)


results = run_inference(model_continuous__query_1, 10000)
print("")
print("Posterior (observational) inference:")
print("ExpectedValue(X | Y = 3)")
print(calculate_expectation(results))
print("***")


###


def model_continuous__query_2():
    x = NormalERP(mean=0, std=1)
    z = NormalERP(mean=0, std=1)
    y = ObservableNormalERP(mean=x.value + z.value, noise_std=2, depends_on=[x, z])
    observe(y, 3)
    do(z, 2)
    predict(y.value)


results = run_inference(model_continuous__query_2, 10000)
print("")
print("Counterfactual inference:")
print("ExpectedValue(Y' | Y = 3, do(Z = 2))")
print(calculate_expectation(results))
print("***")
