
import sys
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import os


INPUT_FOLDER = "output/experiment_results/processed/"
OUTPUT_FOLDER = "output/plots/"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


rc('font',**{'family':'serif', 'size': 14})
rc('text', usetex=True)


def plot_results(filepath, outpath, logscale):
    d = json.load(open(filepath))
    palette = sns.color_palette()
    systems = [
        'Pyro, without guide',
        'Pyro, with guide',
        'MultiVerse, not optimised',
        'MultiVerse, optimised'
    ]
    systems = [s for s in systems if s in d.keys()]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    for i, system in enumerate(systems):
        X = sorted([int(k) for k in d[system].keys()])
        results = [d[system][str(key)] for key in X]
        min_length = min([len(l) for l in results])
        results = [r[:min_length] for r in results]
        M = np.array(results).T
        mu = np.abs(M).mean(axis=0)
        p10 = np.percentile(np.abs(M), 10, interpolation='midpoint', axis=0)
        p90 = np.percentile(np.abs(M), 90, interpolation='midpoint', axis=0)

        c = palette[i]
        ax.fill_between(X, p10, p90, alpha=0.13, zorder=1, color=c)
        if system == 'MultiVerse, optimised':
            zorder = 3
            marker = 'x'
        else:
            zorder = 2
            marker = 'o'
        ax.plot(X, mu, linewidth=2,  label="{}".format(system.replace("MV", "MultiVerse")), zorder=zorder, color=c,
                 marker=marker, markersize=8)
    plt.legend()

    if logscale:
        plt.xscale('log')

    plt.ylabel("Absolute error")
    plt.title("Absolute error across systems vs. Number of samples")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])
    plt.xlabel("Number of samples")
    plt.margins(0.0)
    plt.savefig(outpath)


def plot_time(filepath, outpath):
    d = json.load(open(filepath))
    palette = sns.color_palette()
    sys1 = ['Pyro, without guide', 'Pyro, with guide']
    sys2 = ['MultiVerse, not optimised', 'MultiVerse, optimised']
    sys_collection = [sys1, sys2]
    i = 0
    for k, systems in enumerate(sys_collection):
        plt.figure(figsize=(10, 6))
        for system in systems:
            X = sorted([int(k) for k in d[system].keys()])
            results = [d[system][str(key)] for key in X]
            min_length = min([len(l) for l in results])
            results = [r[:min_length] for r in results]
            M = np.array(results).T
            mu = np.abs(M).mean(axis=0)
            p10 = np.percentile(np.abs(M), 10, interpolation='midpoint', axis=0)
            p90 = np.percentile(np.abs(M), 90, interpolation='midpoint', axis=0)
            c = palette[i]
            i += 1
            plt.fill_between(X, p10, p90, alpha=0.13, zorder=1, color=c)
            if system == 'MultiVerse, optimised':
                zorder = 3
                marker = 'x'
            else:
                zorder = 2
                marker = 'o'
            plt.plot(X, mu, linewidth=2,  label=system, zorder=zorder, color=c,
                     marker=marker, markersize=8)
        plt.legend()
        plt.ylabel("Time to execute (s)")
        plt.title("Execution time across systems vs. Number of samples")
        plt.xlabel("Number of samples")
        plt.margins(0.0)
        plt.savefig(outpath + "execution_time_{}.pdf".format(k))


if __name__ == "__main__":
    plot_results(
        INPUT_FOLDER + "datapoints.json",
        OUTPUT_FOLDER + "accuracy_results.pdf",
        logscale=True,
    )

    plot_time(
        INPUT_FOLDER + "logged_times_per_engine.json",
        OUTPUT_FOLDER
    )

