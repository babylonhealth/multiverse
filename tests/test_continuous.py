
import pytest

from multiverse.engine import (
    NormalERP,
    ObservableNormalERP,
    do,
    observe,
    predict,
    run_inference,
)
from utils import calculate_expectation

# Calculating the approximate ground truth:
#
# import numpy
# import scipy.stats
# predictions = []
# weights = []
# for sample_index in range(100000):
#     x = numpy.random.normal(0, 1)
#     z = numpy.random.normal(0, 1)
#     y_epsilon = 3 - (x + z)
#     weights.append(
#         scipy.stats.norm.pdf(y_epsilon, loc=0, scale=2)
#     )
#     predictions.append(x + 5 + y_epsilon)
# print(numpy.average(predictions, weights=weights))
# > 7.498017852185622


def model_continuous__query_1():
    x = NormalERP(mean=0, std=1)
    z = NormalERP(mean=0, std=1)
    y = ObservableNormalERP(
        mean=x.value + z.value,
        noise_std=2,
        depends_on=[x, z]
    )
    observe(y, 3)
    do(z, 5)
    predict(y.value)


def test_continuous():
    results = run_inference(model_continuous__query_1, 10000)
    assert (
        calculate_expectation(results) ==
        pytest.approx(7.498017852185622, rel=1e-2)
    )
