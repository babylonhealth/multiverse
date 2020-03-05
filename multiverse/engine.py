

import math
import multiprocessing
from datetime import datetime

import numpy
from scipy.stats import norm

DOTYPE_CF = "DOTYPE_CF"
DOTYPE_IV_CF = "DOTYPE_IV_CF"
DOTYPE_IV = "DOTYPE_IV"


EPSILON = 10 ** -10


# Below we define all variables that are used
# for the state of the inference engine.
# To allow the simplest and most straightforward use
# of the engine, they are global variables per this model.
# That does allow a straightforward use of the engine
# (like a user can just import the methods from it)
# but that obviously also limit the user to perform
# only one inference per time per Python interpret.
# In another implementation all this variables can be
# part of the inference engine's state that is passed
# for each method, like:
#
#   engine = create_engine()
#   ...
#   do(engine, z, 2)
#
# or even:
#
#   engine, my_observe, my_do = create_engine(expose_methods=True)
#   ...
#   my_do(z, 2)

TRACE = None
LOG_LIKELIHOOD = None
LOG_PROPOSAL = None

# Note that in the current implementation,
# both POSITIONS_TO_DO_CF and POSITIONS_TO_DO_IV
# are calculated only once during the "zeroth run".
# It works fine if either:
# 1. The order and number of random choices is fixed;
# 2. or, the user of the engine applies their own *consistent*
#    random choice addressing scheme.
# Otherwise, the "do" operation calls won't be consistent.
# In the future, it might be possible to record the locations
# on every run during the "abduction" run step.
POSITIONS_TO_DO_CF = None
POSITIONS_TO_DO_IV = None

PREDICTIONS = None
USING_TRACE_POSITION = None
USED_TRACE_ADDRESSES = None
DONE_VARIABLES_AND_DESCENDANTS = None
OBSERVATIONS = None
NEED_TO_DO_COUNTERFACTUAL = None

# The state of the engine:
#   NOT_INITIALISED
#   ZERO_RUN
#   FIRST_RUN
#   SECOND_RUN
MODE = "NOT_INITIALISED"


def ensure_trace_address_used_only_one(trace_address):
    global USED_TRACE_ADDRESSES
    assert trace_address not in USED_TRACE_ADDRESSES
    USED_TRACE_ADDRESSES.add(trace_address)


class ElementaryRandomProcess:
    def __init__(
        self,
        has_its_own_value=True,
        trace_address=None,
    ):
        if trace_address is None:
            global USING_TRACE_POSITION
            USING_TRACE_POSITION += 1
            self._trace_position = str(USING_TRACE_POSITION)
        else:
            self._trace_position = str(trace_address)

        ensure_trace_address_used_only_one(self._trace_position)

        if has_its_own_value:
            self._value = None

    def determine_value_and_record(self):
        global TRACE
        global LOG_LIKELIHOOD
        global LOG_PROPOSAL
        global POSITIONS_TO_DO_IV

        if self._trace_position in POSITIONS_TO_DO_IV:
            # TODO: generally, it can be in both.
            # We can support it in the future.
            assert self._trace_position not in OBSERVATIONS

            forced_value = POSITIONS_TO_DO_IV[self._trace_position]
            TRACE[self._trace_position] = forced_value
            self._value = forced_value
            LOG_LIKELIHOOD += 0
        elif self._trace_position in OBSERVATIONS:
            observed_value = OBSERVATIONS[self._trace_position]
            TRACE[self._trace_position] = observed_value
            self._value = observed_value
            LOG_LIKELIHOOD += self.log_likelihood()
        else:
            self._value = self.sample(using_proposal=True)
            TRACE[self._trace_position] = self._value
            LOG_LIKELIHOOD += self.log_likelihood()
            LOG_PROPOSAL += self.log_likelihood_proposal()

    def sample(self):
        raise NotImplementedError

    def log_likelihood(self, value):
        raise NotImplementedError

    def observe(self, value):
        global TRACE
        global OBSERVATIONS

        # Here we record it by the object, and later
        # we extract the final address.
        # That is because for observable ERPs
        # it will be the result variable's address.
        OBSERVATIONS[self] = value
        self._value = value
        TRACE[self._trace_position] = value

    def on_erp_init(self, depends_on):
        global TRACE
        global MODE

        if MODE == "ZERO_RUN" or MODE == "FIRST_RUN":
            self.determine_value_and_record()
        elif MODE == "SECOND_RUN":
            global DONE_VARIABLES_AND_DESCENDANTS

            if self._trace_position in DONE_VARIABLES_AND_DESCENDANTS:
                # `do` was applied directly:
                self._value = TRACE[self._trace_position]
            else:
                for ancestor in depends_on:
                    if ancestor.get_trace_position() in DONE_VARIABLES_AND_DESCENDANTS:
                        DONE_VARIABLES_AND_DESCENDANTS.add(self._trace_position)
                        # Note: we are resampling the value.
                        self._value = self.sample(using_proposal=False)
                        break
                else:
                    self._value = TRACE[self._trace_position]

    @property
    def value(self):
        return self._value

    def do(self, value, do_type):
        global POSITIONS_TO_DO_IV
        global POSITIONS_TO_DO_CF

        if MODE == "ZERO_RUN":
            if do_type in [DOTYPE_IV, DOTYPE_IV_CF]:
                POSITIONS_TO_DO_IV[self._trace_position] = value
            if do_type in [DOTYPE_CF, DOTYPE_IV_CF]:
                POSITIONS_TO_DO_CF[self._trace_position] = value
        else:
            pass

    def get_trace_position(self):
        return self._trace_position


def normal_log_pdf_with_edge_case(value, mean, std):
    if std == 0.0:
        if abs(mean - value) < EPSILON:
            return 0.0
        else:
            return float("-inf")
    else:
        return numpy.log(norm.pdf(value, mean, std))


class NormalERP(ElementaryRandomProcess):
    def __init__(
        self,
        mean,
        std,
        proposal_mean=None,
        proposal_std=None,
        depends_on=[],
        trace_address=None,
    ):
        super().__init__(trace_address=trace_address)
        self._mean = mean
        self._std = std
        self._proposal_mean = proposal_mean if proposal_mean is not None else self._mean
        self._proposal_std = proposal_std if proposal_std is not None else self._std
        self.on_erp_init(depends_on=depends_on)

    def sample(self, using_proposal):
        if using_proposal:
            return numpy.random.normal(self._proposal_mean, self._proposal_std, 1)[0]
        else:
            return numpy.random.normal(self._mean, self._std, 1)[0]

    def log_likelihood(self):
        return normal_log_pdf_with_edge_case(
            self._value, self._mean, self._std
        )

    def log_likelihood_proposal(self):
        return normal_log_pdf_with_edge_case(
            self._value, self._proposal_mean, self._proposal_std
        )


def safe_0_log(val):
    if val == 0:
        return float("-inf")
    else:
        return math.log(val)


class BernoulliERP(ElementaryRandomProcess):
    def __init__(
        self,
        prob,
        proposal_prob=None,
        depends_on=[],
        trace_address=None,
    ):
        super().__init__(trace_address=trace_address)
        self._prob = prob
        if proposal_prob is None:
            self._proposal_prob = self._prob
        else:
            self._proposal_prob = proposal_prob
        self.on_erp_init(depends_on=depends_on)

    def sample(self, using_proposal):
        tmp = numpy.random.uniform(0, 1, 1)[0]
        if using_proposal:
            prob = self._proposal_prob
        else:
            prob = self._prob
        if prob == 0.0:
            return 0
        if prob == 1.0:
            return 1
        if tmp < prob:
            return 1
        else:
            return 0

    def log_likelihood(self):
        assert self._value in [0, 1]
        if self._value == 1:
            return safe_0_log(self._prob)
        elif self._value == 0:
            return safe_0_log(1.0 - self._prob)

    def log_likelihood_proposal(self):
        assert self._value in [0, 1]
        if self._value == 1:
            return safe_0_log(self._proposal_prob)
        elif self._value == 0:
            return safe_0_log(1.0 - self._proposal_prob)


class DeltaERP(ElementaryRandomProcess):
    def __init__(
        self,
        delta_value,
        depends_on=[],
        trace_address=None,
    ):
        super().__init__(trace_address=trace_address)
        self._delta_value = delta_value
        self.on_erp_init(depends_on=depends_on)

    def sample(self, using_proposal):
        return self._delta_value

    def log_likelihood(self):
        if self._value == self._delta_value:
            return 0.0
        else:
            return float("-inf")

    def log_likelihood_proposal(self):
        return self.log_likelihood()


class StrictObserveDeltaERP(ElementaryRandomProcess):
    def __init__(
        self,
        delta_value,
        depends_on=[],
        trace_address=None,
    ):
        super().__init__(trace_address=trace_address)
        self._delta_value = delta_value
        self.on_erp_init(depends_on=depends_on)

    def sample(self, using_proposal):
        return self._delta_value

    def log_likelihood(self):
        if (
            self._value == self._delta_value
            or (
                type(self._value) in [float, int] and
                abs(self._value - self._delta_value) < EPSILON
            )
        ):
            return 0.0
        else:
            if MODE != "ZERO_RUN":
                assert False, "StrictObserveDeltaERP does not match the value"
            else:
                # We ignore observations during this run.
                return float("-inf")

    def log_likelihood_proposal(self):
        return self.log_likelihood()


class ObservableERP(ElementaryRandomProcess):
    def __init__(
        self,
        trace_address=None,
    ):
        super().__init__(
            has_its_own_value=False,
            trace_address=trace_address,
        )

    def if_observed(self):
        observed = False
        observed_value = None
        if MODE == "FIRST_RUN":
            result_trace_position = self.get_result_var_trace_address()
            if result_trace_position in OBSERVATIONS:
                observed = True
                observed_value = OBSERVATIONS[result_trace_position]
        return observed, observed_value

    @property
    def value(self):
        return self._result_var.value

    @property
    def noise_value(self):
        return self._noise_var.value

    def observe(self, value):
        self._result_var.observe(value)

    def do(self, value, do_type):
        self._result_var.do(value, do_type)

    def inverse_function(self, hyperparameters, observed_value):
        raise NotImplementedError

    def _extract_noise_var(self):
        return self._noise_var

    def get_noise_var_trace_address(self):
        # Note that the address scheme defined
        # by the user should not use ":".
        return self._trace_position + ":::noise_var"

    def get_result_var_trace_address(self):
        # Note that the address scheme defined
        # by the user should not use ":".
        return self._trace_position + ":::result_var"

    def get_trace_position(self):
        return self.get_result_var_trace_address()


class ObservableNormalERP(ObservableERP):
    def __init__(
        self,
        mean,
        noise_std,
        depends_on=[],
        noise_mean=0,
        noise_depends_on=[],
        trace_address=None,
    ):
        super().__init__(
            trace_address=trace_address,
        )

        observed, observed_value = self.if_observed()

        if observed:
            proposal_mean = self.inverse_function(mean, observed_value)
            proposal_std = 0.0
        else:
            proposal_mean = None
            proposal_std = None

        self._noise_var = NormalERP(
            mean=noise_mean,
            std=noise_std,
            proposal_mean=proposal_mean,
            proposal_std=proposal_std,
            depends_on=noise_depends_on,
            trace_address=self.get_noise_var_trace_address(),
        )

        self._result_var = StrictObserveDeltaERP(
            delta_value=mean + self._noise_var.value,
            depends_on=depends_on + [self._noise_var],
            trace_address=self.get_result_var_trace_address(),
        )

    def inverse_function(self, mean, observed_value):
        return observed_value - mean


class ObservableBernoulliERP(ObservableERP):
    def __init__(
        self,
        input_val,
        noise_flip_prob,
        depends_on=[],
        noise_depends_on=[],
        trace_address=None,
    ):
        super().__init__(
            trace_address=trace_address,
        )

        observed, observed_value = self.if_observed()

        assert input_val in [0, 1]

        if observed:
            assert observed_value in [0, 1]
            if input_val == observed_value:
                proposal_prob = 0.0
            else:
                proposal_prob = 1.0
        else:
            proposal_prob = None

        self._noise_var = BernoulliERP(
            prob=noise_flip_prob,
            proposal_prob=proposal_prob,
            depends_on=noise_depends_on,
            trace_address=self.get_noise_var_trace_address(),
        )

        if self._noise_var.value == 0:
            delta_value = input_val
        else:
            delta_value = 0 if input_val else 1

        self._result_var = StrictObserveDeltaERP(
            delta_value=delta_value,
            depends_on=depends_on + [self._noise_var],
            trace_address=self.get_result_var_trace_address(),
        )


def observe(erp, value):
    # NOTE: for now we don't do counterfactual conditioning,
    #       but we can do it in the future.

    if MODE == "ZERO_RUN":
        erp.observe(value)
    else:
        pass


def do(erp, value, do_type=DOTYPE_CF):
    erp.do(value, do_type)


def predict(expression_value, predict_counterfactual=True):
    global PREDICTIONS
    global NEED_TO_DO_COUNTERFACTUAL
    if MODE == "ZERO_RUN":
        if predict_counterfactual is True:
            NEED_TO_DO_COUNTERFACTUAL = True
    if MODE == "FIRST_RUN" and predict_counterfactual is False:
        # Predict for an observational/interventional query
        PREDICTIONS += [expression_value]
    elif MODE == "SECOND_RUN" and predict_counterfactual is True:
        # Predict for a counterfactual query
        PREDICTIONS += [expression_value]


def run_inference_helper(model_function, num_samples, random_seed):
    numpy.random.seed(datetime.now().microsecond + random_seed)

    global MODE
    global LOG_LIKELIHOOD
    global LOG_PROPOSAL
    global TRACE
    global POSITIONS_TO_DO_CF
    global POSITIONS_TO_DO_IV
    global PREDICTIONS
    global USING_TRACE_POSITION
    global USED_TRACE_ADDRESSES
    global DONE_VARIABLES_AND_DESCENDANTS
    global OBSERVATIONS
    global NEED_TO_DO_COUNTERFACTUAL

    # We need to collect the trace positions and values of
    # `do` and `observe` statements.
    MODE = "ZERO_RUN"

    LOG_LIKELIHOOD = 0.0
    LOG_PROPOSAL = 0.0
    TRACE = {}
    POSITIONS_TO_DO_CF = {}
    POSITIONS_TO_DO_IV = {}
    PREDICTIONS = []
    USING_TRACE_POSITION = -1
    USED_TRACE_ADDRESSES = set()
    DONE_VARIABLES_AND_DESCENDANTS = None
    OBSERVATIONS = {}
    NEED_TO_DO_COUNTERFACTUAL = False

    # Doing the zeroth run to record positions and values
    # for `do` and `observe`.
    model_function()

    # Record appropriate addresses for the observations.
    TMP_OBSERVATIONS = OBSERVATIONS
    OBSERVATIONS = {}
    for class_obj, val in TMP_OBSERVATIONS.items():
        OBSERVATIONS[class_obj.get_trace_position()] = val

    # Now we do the "regular" runs of the importance sampling
    # to record the traces and the log-likelihoods.
    MODE = "FIRST_RUN"

    RUNS = []

    for sample_index in range(num_samples):
        LOG_LIKELIHOOD = 0.0
        LOG_PROPOSAL = 0.0
        USING_TRACE_POSITION = -1
        USED_TRACE_ADDRESSES = set()
        TRACE = {}
        PREDICTIONS = []

        model_function()

        RUNS += [
            {
                "TRACE": TRACE,
                "LOGWEIGHT": LOG_LIKELIHOOD - LOG_PROPOSAL,
                "PREDICTIONS": PREDICTIONS,
            }
        ]

    # If we have any counterfactual internvetions and
    # predictions, we also do them.
    if NEED_TO_DO_COUNTERFACTUAL:
        MODE = "SECOND_RUN"

        for run_info in RUNS:
            USING_TRACE_POSITION = -1
            USED_TRACE_ADDRESSES = set()

            DONE_VARIABLES_AND_DESCENDANTS = set()

            # Replacing the values for variables, for which
            # "counter-factual" `do` was applied.
            for position, value in POSITIONS_TO_DO_CF.items():
                run_info['TRACE'][position] = value
                DONE_VARIABLES_AND_DESCENDANTS.add(position)

            TRACE = run_info['TRACE']

            PREDICTIONS = run_info['PREDICTIONS']

            model_function()

            run_info['PREDICTIONS'] = PREDICTIONS

            # We don't need the trace anymore.
            run_info.pop("TRACE")

    # All done.
    MODE = "NOT_INITIALISED"

    return RUNS


def run_inference(model_function, num_samples, num_cores=None):
    # TODO:
    # * Instead of running N/C samples in first mode
    #   and then in second mode, run each sample in first
    #   and sample mode to get the better memory complexity
    #   if we only store the PREDICTION-s.

    if num_cores is None:
        pool_size = multiprocessing.cpu_count()
    else:
        pool_size = num_cores

    num_samples_pre_process = math.ceil(num_samples / pool_size)

    pool = multiprocessing.Pool(pool_size)

    sum_num_samples_to_run = 0

    # Simple arithmetic to do binning:
    args = []
    for index in range(pool_size):
        sum_num_samples_to_run += num_samples_pre_process
        if sum_num_samples_to_run >= num_samples:
            num_samples_pre_process = (
                num_samples - (sum_num_samples_to_run - num_samples_pre_process)
            )
            sum_num_samples_to_run = num_samples

        if num_samples_pre_process > 0:
            args += [
                (model_function, num_samples_pre_process, index)
            ]

    assert sum_num_samples_to_run == num_samples

    MULTIPLE_RUNS = pool.starmap(run_inference_helper, args)

    RUNS = []
    for el in MULTIPLE_RUNS:
        RUNS += el

    pool.close()

    assert len(RUNS) == num_samples

    return RUNS


def IF_OBSERVE_BLOCK():
    return MODE != "SECOND_RUN"


def IF_DO_BLOCK(do_type=DOTYPE_CF):
    # Currently it does not matter
    # what `do_type` we are in.
    # They are all done in ZERO_RUN.

    return MODE == "ZERO_RUN"


def compute_procedure_if_necessary(procedure_trace_address, procedure_caller):
    if MODE == "SECOND_RUN" and procedure_trace_address in POSITIONS_TO_DO_CF:
        # Because it has been intervened, the PP type and value
        # do not matter anymore.

        return DeltaERP(
            delta_value=None,
            depends_on=[],
            trace_address=procedure_trace_address,
        )
    else:
        return procedure_caller()
