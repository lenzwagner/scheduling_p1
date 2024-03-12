import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import itertools
import plotly.graph_objects as go
import gurobi_logtools as glt

def combine_legends(*axes):
    handles = list(itertools.chain(*[ax.get_legend_handles_labels()[0] for ax in axes]))
    labels = list(
        itertools.chain(*[ax3.get_legend_handles_labels()[1] for ax3 in axes])
    )
    return handles, labels


def set_obj_axes_labels(ax):
    ax.set_ylabel("Objective value")
    ax.set_xlabel("Time")


def plot_incumbent(df, ax):
    ax.step(
        df["Time"],
        df["Incumbent"],
        where="post",
        color="b",
        label="Primal Bound",
    )
    set_obj_axes_labels(ax)


def plot_bestbd(df, ax):
    ax.step(
        df["Time"],
        df["BestBd"],
        where="post",
        color="r",
        label="Dual Bound",
    )
    set_obj_axes_labels(ax)


def plot_fillabsgap(df, ax):
    ax.fill_between(
        df["Time"],
        df["BestBd"],
        df["Incumbent"],
        step="post",
        color="grey",
        alpha=0.3,
    )
    set_obj_axes_labels(ax)


def plot_relgap(df, ax):
    ax.step(
        df["Time"],
        df["Gap"],
        where="post",
        color="green",
        label="Gap",
    )
    ax.set_ylabel("Optimality Gap in %")
    ax.set_ylim(0, 1)
    formatter = PercentFormatter(1)
    ax.yaxis.set_major_formatter(formatter)


def plot(df, time):
    with plt.style.context("seaborn-v0_8"):
        _, ax = plt.subplots(figsize=(8, 5))

        plot_incumbent(df, ax)
        plot_bestbd(df, ax)
        plot_fillabsgap(df, ax)

        ax2 = ax.twinx()
        plot_relgap(df, ax2)

        ax.set_xlim(1, time)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        print(combine_legends(ax, ax2))
        ax.legend(*combine_legends(ax, ax2))

        plt.show()

def parse_log(path1):
    path = path1
    results, timeline = glt.get_dataframe([path], timelines=True)
    default_run = timeline["nodelog"]

    return default_run