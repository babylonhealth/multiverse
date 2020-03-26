
import json
from collections import defaultdict
import numpy
import os


INPUT_FOLDER = "output/experiment_results/"
OUTPUT_FOLDER = INPUT_FOLDER + "processed/"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


DATAPOINTS = defaultdict(lambda: defaultdict(list))

LOGGED_TIMES_PER_ENGINE = defaultdict(lambda: defaultdict(list))


all_experiment_runs = json.load(open(INPUT_FOLDER + "experiments.json"))

for experiment_run in all_experiment_runs:
    num_samples = experiment_run['num_samples']
    algorithm_name = experiment_run['algorithm_name']
    result = experiment_run['result']
    ground_truth_value = experiment_run['ground_truth_value']
    logged_time = experiment_run['logged_time']

    DATAPOINTS[algorithm_name][num_samples].append(
        abs(result - ground_truth_value)
    )

    LOGGED_TIMES_PER_ENGINE[algorithm_name][num_samples].append(
        logged_time
    )

json.dump(
    DATAPOINTS,
    open(OUTPUT_FOLDER + "datapoints.json", "w"),
    indent=2,
)

json.dump(
    LOGGED_TIMES_PER_ENGINE,
    open(OUTPUT_FOLDER + "logged_times_per_engine.json", "w"),
    indent=2,
)
