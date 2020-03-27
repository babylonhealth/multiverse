
import sys
sys.path.append("..")

from multiverse.engine import (
    BernoulliERP,
    DeltaERP,
    do,
    observe,
    predict,
    run_inference,
)
from utils import calculate_expectation

# This example is based on the paper ``Counterfactual diagnosis''
# by Jonathan G. Richens, Ciaran M. Lee, Saurabh Johri
# URL: https://arxiv.org/abs/1910.06772


# Note: this is not the most efficient way to implement the noisy-OR
# with observations. That is, here we just do the rejection sampling.
# For more info, please, refer to https://arxiv.org/abs/1910.08091,
# section E.7 ``More sophisticated example of an Observable ERP:
# the Observable Noisy OR''.
def noisy_or_value(leak_0_outcome, disease_states, disease_leak_outcomes):
    symptom_delta_value = 0
    assert len(disease_states) == len(disease_leak_outcomes)
    if leak_0_outcome == 1:
        symptom_delta_value = 1
    for index, disease_state in enumerate(disease_states):
        if disease_state == 1 and disease_leak_outcomes[index] == 1:
            symptom_delta_value = 1
    return symptom_delta_value


def model_discrete_uncollapsed():
    risk_factor_alpha = BernoulliERP(prob=0.01, proposal_prob=0.2)
    disease_alpha = BernoulliERP(
        prob=0.2 if risk_factor_alpha.value else 0.000001,
        proposal_prob=0.2,
        depends_on=[risk_factor_alpha]
    )
    disease_beta = BernoulliERP(
        prob=0.2 if risk_factor_alpha.value else 0.000001,
        proposal_prob=0.2,
        depends_on=[risk_factor_alpha]
    )
    disease_gamma = BernoulliERP(
        prob=0.95 if risk_factor_alpha.value else 0.0001,
        proposal_prob=0.1,
        depends_on=[risk_factor_alpha]
    )
    leak_0 = BernoulliERP(prob=0.0001, proposal_prob=0.1)
    leak_alpha = BernoulliERP(prob=0.5)
    leak_beta = BernoulliERP(prob=0.5)
    symptom_delta_value = noisy_or_value(
        leak_0.value,
        [disease_alpha.value, disease_beta.value],
        [leak_alpha.value, leak_beta.value],
    )
    symptom = DeltaERP(
        delta_value=symptom_delta_value,
        depends_on=[disease_alpha, disease_beta, leak_0, leak_alpha, leak_beta],
    )
    return disease_alpha, disease_beta, disease_gamma, symptom


def model_discrete_uncollapsed__query_1():
    disease_alpha, disease_beta, disease_gamma, symptom = model_discrete_uncollapsed()
    observe(symptom, 1)
    predict(disease_alpha.value)
    predict(disease_beta.value)
    predict(disease_gamma.value)


def model_discrete_uncollapsed__query_2():
    disease_alpha, disease_beta, disease_gamma, symptom = model_discrete_uncollapsed()
    observe(symptom, 1)
    do(disease_alpha, 0)
    predict(1 - symptom.value)


def model_discrete_uncollapsed__query_3():
    disease_alpha, disease_beta, disease_gamma, symptom = model_discrete_uncollapsed()
    observe(symptom, 1)
    do(disease_beta, 0)
    predict(1 - symptom.value)


def model_discrete_uncollapsed__query_4():
    disease_alpha, disease_beta, disease_gamma, symptom = model_discrete_uncollapsed()
    observe(symptom, 1)
    do(disease_gamma, 0)
    predict(1 - symptom.value)


print("""
This example is based on the paper ``Counterfactual diagnosis''
by Jonathan G. Richens, Ciaran M. Lee, Saurabh Johri
URL: https://arxiv.org/abs/1910.06772

Let's say there is a risk factor RF1 that significantly increases
the probabilities of Disease D1, Disease D2 and Disease D3. That is,
it is likely to have any of Diseases only if RF1 is True.

Let's say that:
   RF1 ~ Bernoulli(0.01)
   D1  ~ Bernoulli(0.2  if RF1 else 0.000001)
   D2  ~ Bernoulli(0.2  if RF1 else 0.000001)
   D3  ~ Bernoulli(0.95 if RF1 else 0.0001)

Now, let's say that Symptom is caused only by Disease 1 and
Disease 2 (but not Disease 3) as follows:
   Symptom = OR(
       Disease1 && BernoulliLeak(probTrue = 0.5)
       OR Disease2 && BernoulliLeak(probTrue = 0.5)
       OR BernoulliSymptomLeak(probTrue = 0.0001)
   )

Now, let's calculate posterior probabilities, in other words
how likely that Disease X is True if the Symptom is True:
""")


results = run_inference(model_discrete_uncollapsed__query_1, 1000000)
print("  ", "P(D1 = T | S = T)")
print("  ", calculate_expectation(results, prediction_index=0))
print("  ", "***")
print("  ", "P(D2 = T | S = T)")
print("  ", calculate_expectation(results, prediction_index=1))
print("  ", "***")
print("  ", "P(D3 = T | S = T)")
print("  ", calculate_expectation(results, prediction_index=2))

print("""

You can see that the marginal posterior P(D_i = T | S = T) for each
of three diseases is high, even though Disease 3 does not cause
the symptom at all. This is due to the back-door path through the
risk factor. Now, let's calculate the counter-factual query of

  "How likely that the Symptom would be treated (i.e. False) in a patient
   if initially that patient had had that Symptom and we cured
   Disease X for sure."

""")


results = run_inference(model_discrete_uncollapsed__query_2, 1000000)
print("  ", "P(S' = F | S = T, do(D1 = F))")
print("  ", calculate_expectation(results))
print("  ", "***")


results = run_inference(model_discrete_uncollapsed__query_3, 1000000)
print("  ", "P(S' = F | S = T, do(D2 = F))")
print("  ", calculate_expectation(results))
print("  ", "***")


results = run_inference(model_discrete_uncollapsed__query_4, 1000000)
print("  ", "P(S' = F | S = T, do(D3 = F))")
print("  ", calculate_expectation(results))


print("""

You can see that if we treated Disease 1 or Disease 2, the Symptom would
very likely disappear (i.e. to switch to False), but if we treated
Disease 3, it is absolutely unlikely (with probability 0.0) that
the Symptom would disappear.

Hence, Disease 3 is definitely not the cause of Symptom, while
Disease 1 and Disease 2 might be the cause.

Note that the alternative (for the posterior inference) might be to only
consider the posterior values of diseases that are connected to positive
symptoms and potentially filter out the other diseases. Also, care must
be taken of incidences of diseases: appropriate probability measures
should be used with appropriate modelling for those random variables.

Also note that with multiple symptoms and if diseases are causing different
number of symptoms, the counterfactual measure of _how many symptoms
would be treated_ needs to be adjusted in a non-trivial way to account
for that.
""")
