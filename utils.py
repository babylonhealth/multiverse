
try:
    from scipy.misc import logsumexp
except ImportError:
    from scipy.special import logsumexp
import numpy


def calculate_expectation(results, prediction_index=0):
    values = [result['PREDICTIONS'][prediction_index] for result in results]
    logweights = [result['LOGWEIGHT'] for result in results]
    values_by_logweights, sign = logsumexp(a=logweights, b=values, return_sign=True)
    logsum_weights = logsumexp(a=logweights)
    expectation = numpy.exp(values_by_logweights - logsum_weights)
    if sign == 0.0:
        return 0.0
    elif sign == -1.0:
        return expectation * -1.0
    else:
        assert sign == 1.0
        return expectation


def calculate_ess(results):
    logweights = [result['LOGWEIGHT'] for result in results]
    logsum_weights_squared = 2.0 * logsumexp(a=logweights)
    logsum_squared_weights = logsumexp(a=[2.0 * val for val in logweights])

    ESS = numpy.exp(
        logsum_weights_squared - logsum_squared_weights
    )

    # enum = numpy.power(sum([numpy.exp(logw) for logw in logweights]), 2.0)
    # denom = sum([numpy.power(numpy.exp(logw), 2.0) for logw in logweights])
    # ESS = enum / denom

    return ESS
