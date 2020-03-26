
import json
import os
from tqdm import tqdm

from compute_pyro import run_pyro, AllSamplesRejectedException
from compute_mv import run_mv_non_opt, run_mv_opt


OUTPUT_FOLDER = "output/experiment_results/"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def run_pyro_with_guide(experiment_id, num_samples):
    return run_pyro(experiment_id, num_samples, type_to_run="pyro_with_guide")

def run_pyro_without_guide(experiment_id, num_samples):
    while True:
        try:
            return run_pyro(experiment_id, num_samples, type_to_run="pyro_without_guide")
            break
        except AllSamplesRejectedException:
            continue


# Settings of the experiments:
experiment_ids_to_run = range(20)
num_trials = 10
num_samples_options = [25, 50, 100, 200, 500, 1000]


algorithms = [
    ("MultiVerse, not optimised", run_mv_non_opt),
    ("MultiVerse, optimised", run_mv_opt),
    ("Pyro, without guide", run_pyro_without_guide),
    ("Pyro, with guide", run_pyro_with_guide),
]
algorithms = dict(algorithms)

ground_truth_values = {}
for experiment_id in experiment_ids_to_run:
    ground_truth_values[experiment_id] = json.load(
        open("output/calculated_gts/" + str(experiment_id) + ".json")
    )

all_experiments = []

for experiment_id in experiment_ids_to_run:
    for num_samples in num_samples_options:
        for trial_id in range(num_trials):
            for algorithm_name in algorithms.keys():
                all_experiments.append(
                    {
                        "experiment_id": experiment_id,
                        "ground_truth_value": ground_truth_values[experiment_id],
                        "num_samples": num_samples,
                        "trial_id": trial_id,
                        "algorithm_name": algorithm_name,
                    }
                )

for experiment_run in tqdm(all_experiments):
    algorithm_function = algorithms[experiment_run['algorithm_name']]
    result, time = algorithm_function(
        experiment_id=experiment_run['experiment_id'],
        num_samples=experiment_run['num_samples'],
    )
    experiment_run['result'] = result
    experiment_run['logged_time'] = time

json.dump(
    all_experiments,
    open(OUTPUT_FOLDER + "experiments.json", "w"),
    indent=2,
)

