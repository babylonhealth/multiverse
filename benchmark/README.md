
# Introduction

This is a benchmark for MultiVerse on some number of synthetically generated Bayesian networks.

We also implement it with Pyro for comparison.

It follows the experiments in the paper (see parent folder's `README.md` for more details).



# Synthetic Networks

A number of synthetic (acyclic) Bayesian networks was generated. They are located in `experiment_inputs/`.


## Network Structure

There are exogenous Bernoulli nodes with a random prior.

There are also endogenous nodes. Each endogenous node `EndogNode` has parents p1, ..., pN (which are other nodes in the network) and associated random weights w_1, ..., w_N (which are positive and sum to 1.0), and it also has its own Bernoulli noise variable `NoisyFlipper` with a random prior. The value of the endogenous node is determined as follows:

  `Inverse(val) = 0.0 if val else 1.0`
  
  `InverseIf(val, predicate) = Inverse(val) if predicate else val`
  
  `AF(p1, ..., pN) = 1.0 if Sum(w_i * p_i) > 0.5 else 0.0`
  
  `EndogNode = InverseIf(AF(p1, ..., pN), NoisyFlipper)`

There are 15 nodes in each network that are either exogenous (but not `NoisyFlipper`-s) or endogenous.



# How to run

1. Install extra requirements: `pip install -r extra_requirements.txt`.

2. Calculate ground truth values for the experiments: `python compute_gt.py`

3. Check the settings of the experiment in `runner.py`. Run it.

4. Run `process_results.py` to do simple post-processing steps over the results.

5. Run `plot.py` to generate plots.

6. Check the plots in `output/` folder.

Note that by default only one core is used for MultiVerse experiments. You can change the number of cores to use via variable `MULTIVERSE_NUM_CORES_TO_USE` in `config.py`.



# Extra

* You can visualise any network by running `python visualise.py --exp_id ID` where ID is an experiment ID. It outputs a graph into `output/` folder. For it, you need to install `pygraphviz` as well (e.g. via `pip`).

* Folder `example_precomputed_results/` contains examples of runs of experiments with 1 and 2 cores for MultiVerse.

