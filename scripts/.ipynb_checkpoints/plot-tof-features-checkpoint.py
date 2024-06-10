import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from pydmclab.plotting.utils import set_rc_params

DATA_DIR = "/Users/noord014/MyDrive/phd/research/dmc-lab/bartel-group/programmable-catalysis/dev/data/csvs"
FIG_DIR = "/Users/noord014/MyDrive/phd/research/dmc-lab/bartel-group/programmable-catalysis/dev/figures/"

"""
Fix some default plotting parameters
This wont actually run if you dont have access to our groups package..
Will need to comment it out
"""
set_rc_params()


def load_data(filename, data_dir=DATA_DIR):
    """
    Args:
        filename (string): The name of the file to load.
        data_dir (string, optional): The name of the directory to load from. Defaults to DATA_DIR.

    Returns:
        pandas.DataFrame
    """
    return pd.read_csv(os.path.join(data_dir, filename))


def plot_TOF_histograms(
    data_dict,
    labels,
    bins=100,
    ax_params=None,
    filename="TOF-Histograms.png",
    fig_dir=FIG_DIR,
):
    """
    Plotting as stacked bars since the data contains both positive and negative values.
    Negatives must be plotted as absolute values since the log scale cannot handle negative values.

    Args:
        data_dict (dict): A dictionary mapping histogram titles to the associated data.
        bins (int, optional):  The number of bins to use in the histogram. Defaults to 100.
        labels (list): A list of labels for the histogram bars.
        ax_params (dict, optional): A dictionary mapping histogram titles to the associated axis parameters. Defaults to None.

    Returns:
        None
    """

    fig, ax = plt.subplots(1, len(data_dict), figsize=(18, 6), sharey=True)
    cmap = plt.get_cmap("prism")
    colors = cmap(np.linspace(0, 1, len(labels)))
    data_items = data_dict.items()

    if len(data_dict) == 1:
        ax = [ax]

    for i, (title, data) in enumerate(data_items):
        ax[i].hist(
            data,
            bins=bins,
            histtype="barstacked",
            edgecolor="black",
            color=colors[: len(labels)],
            linewidth=1.5,
            label=labels,
        )
        ax[i].set(**ax_params[title])
        ax[i].set_title(title, pad=12)
        ax[i].yaxis.set_tick_params(labelleft=True, labelright=False)
        ax[i].yaxis.set_minor_locator(AutoMinorLocator(2))
        ax[i].tick_params(
            which="minor",
            axis="both",
            direction="out",
            top=True,
            right=True,
            length=5,
            width=1,
        )

    plt.legend(loc="upper right", bbox_to_anchor=(1, 1))
    plt.show()
    fig.savefig(os.path.join(fig_dir, filename), dpi=300, bbox_inches="tight")

    return None


def plot_feature_histograms(
    feature_data,
    bins=100,
    ax_params=None,
    filename="Feature-Histograms.png",
    fig_dir=FIG_DIR,
):
    """
    Function for plotting histograms of the feature data.

    Args:
        feature_data (dict): A dictionary mapping feature names to the associated data.
        bins (int, optional): The number of bins to use in the histogram. Defaults to 100.
        ax_params (dict, optional): A dictionary mapping feature names to the associated axis parameters. Defaults to None.

    Returns:
        None
    """

    fig, ax = plt.subplots(
        math.ceil(len(feature_data.keys()) / 3), 3, figsize=(36, 28), sharey=True
    )

    for i, (feature_name, data) in enumerate(feature_data.items()):
        ax[i // 3, i % 3].hist(
            data,
            bins=bins,
            histtype="bar",
            edgecolor="black",
            color=["blue"],
            linewidth=1.5,
        )
        ax[i // 3, i % 3].set(**ax_params[feature_name])
        if i % 3 == 0:
            ax[i // 3, i % 3].set_ylabel("Counts", fontsize=24)
        ax[i // 3, i % 3].yaxis.set_minor_locator(AutoMinorLocator(2))
        ax[i // 3, i % 3].xaxis.set_minor_locator(AutoMinorLocator(2))
        ax[i // 3, i % 3].xaxis.get_label().set_fontsize(24)
        ax[i // 3, i % 3].tick_params(
            which="minor",
            axis="both",
            direction="out",
            top=True,
            right=True,
            length=5,
            width=1,
        )
        ax[i // 3, i % 3].set_xlim(
            np.nanmin(data) - 0.1 * np.nanmin(data),
            np.nanmax(data) + 0.1 * np.nanmax(data),
        )

    plt.show()
    fig.savefig(os.path.join(fig_dir, filename), dpi=300, bbox_inches="tight")
    return None


def main():

    filenames = ["240219_op_steady.csv", "240219_rc_steady.csv"]
    tags = ["op", "rc"]

    zipped_specs = zip(filenames, tags)

    for filename, tag in zipped_specs:

        df = load_data(filename)

        TOF_histogram_params = {
            "Steady State TOF": {
                "xlabel": "Loop TOF [1/s]",
                "ylabel": "Bin Count",
                "xscale": "log",
                "xlim": (1e-6, 1e2),
            },
        }

        TOF_histogram_data = {
            "Steady State TOF": [
                df["loop-tof"].loc[df["loop-tof"] > 0],
                df["loop-tof"].loc[df["loop-tof"] < 0].abs(),
            ],
        }

        print("Positive TOF:", len(TOF_histogram_data["Steady State TOF"][0]))
        print("Negative TOF:", len(TOF_histogram_data["Steady State TOF"][1]))

        plot_TOF_histograms(
            TOF_histogram_data,
            labels=["Positive", "Negative"],
            bins=np.logspace(-6, 2, 50),
            ax_params=TOF_histogram_params,
            filename=f"TOF-Histograms-{tag}.png",
        )

        feature_data = {
            feature_name: df[feature_name].values
            for feature_name in df.columns
            if feature_name not in ["steady-state-condition", "loop-tof"]
        }

        feature_histogram_params = {
            feature_name: {
                "xlabel": feature_name,
            }
            for feature_name in feature_data.keys()
        }

        plot_feature_histograms(
            feature_data,
            bins=np.logspace(-5, 3, 1000),
            ax_params=feature_histogram_params,
            filename=f"Feature-Histograms-{tag}.png",
        )

    return None


if __name__ == "__main__":
    main()
