import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CSVS_DIR = os.path.join("..", "..", "data", "csvs")
FIGS_DIR = os.path.join("..", "..", "figures")


def set_rc_params():
    """
    Args:

    Returns:
        dictionary of settings for mpl.rcParams
    """
    params = {
        "axes.linewidth": 2,
        "axes.unicode_minus": False,
        "figure.dpi": 300,
        "font.size": 26,
        "font.family": "arial",
        "legend.frameon": False,
        "legend.handletextpad": 0.4,
        "legend.handlelength": 1,
        "legend.fontsize": 22,
        "mathtext.default": "regular",
        "savefig.bbox": "tight",
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "xtick.major.width": 2,
        "ytick.major.width": 2,
        "xtick.top": True,
        "ytick.right": True,
        "axes.edgecolor": "black",
        "figure.figsize": [6, 4],
    }
    for p in params:
        mpl.rcParams[p] = params[p]
    return params


def load_csv(filename, data_dir=CSVS_DIR):
    """
    Args:
        filename (string): The name of the file to load. (.csv)
        data_dir (string, optional): The name of the directory to load from. Defaults to CSVS_DIR.

    Returns:
        pandas.DataFrame
    """
    return pd.read_csv(os.path.join(data_dir, filename))


def split_TOFs_by_sign(df):
    """
    Args:
        df (pandas.DataFrame): The DataFrame to split

    Returns:
        pandas.DataFrame, pandas.DataFrame, pandas.DataFrame: DataFrames with positive, negative, and zero TOFs
    """
    positive_df = df[df["loop-tof"] > 0]
    negative_df = df[df["loop-tof"] < 0]
    zero_df = df[df["loop-tof"] == 0]
    return positive_df, negative_df, zero_df


def split_TOFs_by_BEA(df):
    """
    Args:
        df (pandas.DataFrame): The DataFrame to split

    Returns:
        pandas.DataFrame, pandas.DataFrame, pandas.DataFrame: DataFrames with low, medium, and high BEA values
    """
    low_bea_df = df[df["change-in-bea"] == 0.3]
    med_bea_df = df[df["change-in-bea"] == 0.5]
    high_bea_df = df[df["change-in-bea"] == 0.8]
    return low_bea_df, med_bea_df, high_bea_df


def plot_piechart(df, filename="TOF_piechart.png"):
    """
    Args:
        df (pandas.DataFrame): The DataFrame to plot
    """
    pos_df, neg_df, zero_df = split_TOFs_by_sign(df)

    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        [len(pos_df), len(neg_df), len(zero_df)],
        explode=(0.06, 0.06, 0),
        shadow=True,
        autopct="%.0f%%",
        pctdistance=0.6,
        startangle=90,
        textprops={
            "color": "black",
            "weight": "bold",
            "fontsize": 18,
        },
        wedgeprops={"edgecolor": "black", "linewidth": 1.5, "antialiased": True},
        colors=["#2f4b7c", "#ffa600", "#bc5090"],
    )

    ax.legend(
        wedges,
        ["Positive", "Negative", "Zero"],
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
    )

    plt.savefig(os.path.join(FIGS_DIR, "figure2", filename), dpi=300)
    plt.show()

    return fig, ax


def plot_histogram(df, filename="TOF_histogram.png"):
    """
    Args:
        df (pandas.DataFrame): The DataFrame to plot
    """

    pos_df, neg_df, zero_df = split_TOFs_by_sign(df)
    bins = np.logspace(np.log10(1e-4), np.log10(1e2), 151)

    fig, ax = plt.subplots(2, 1, figsize=(14, 10), dpi=300, gridspec_kw={"hspace": 0.0})

    ax[0].hist(
        pos_df["loop-tof"], bins, color="#2f4b7c", linewidth=1.0, edgecolor="black"
    )
    ax[0].legend(labels=["Positive"], fontsize=28, loc="upper right", frameon=False)

    ax[1].hist(
        -neg_df["loop-tof"], bins, color="#ffa600", linewidth=1.0, edgecolor="black"
    )
    ax[1].legend(labels=["Negative"], fontsize=28, loc="lower right", frameon=False)

    for k in range(2):
        ax[k].set_xlim([5 * 1e-5, 1e2])
        ax[k].set_xscale("log")
        ax[k].set_yscale("log")

    fig.text(
        0.025,
        0.5,
        "Number of occurences",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=28,
    )

    ax[0].tick_params(
        which="major",
        axis="both",
        direction="out",
        right=True,
        top=True,
        bottom=False,
        length=6,
        width=1.5,
    )
    ax[0].tick_params(
        which="minor",
        axis="both",
        direction="out",
        right=False,
        top=False,
        length=4,
        width=1.5,
    )
    ax[0].set_xticklabels([])
    ax[0].set_ylim([1, 1e4])

    ax[1].set_ylim([1, 1e4])
    ax[1].invert_yaxis()
    ax[1].set_xlabel(f"Loop TOF " r"$(s^{-1})$", fontsize=28)
    ax[1].tick_params(
        which="major",
        axis="both",
        direction="out",
        right=True,
        top=False,
        length=6,
        width=1.5,
    )
    ax[1].tick_params(
        which="minor",
        axis="both",
        direction="out",
        right=False,
        top=False,
        length=4,
        width=1.5,
    )

    plt.savefig(os.path.join(FIGS_DIR, "figure2", filename), dpi=300)
    plt.show()

    return fig, ax


def plot_BEA_histogram(df, filename="BEA_histogram.png"):

    low_bea_df, med_bea_df, high_bea_df = split_TOFs_by_BEA(df)

    bins = np.linspace(-52, 52, 26)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10), dpi=300)
    colors = ["#2A24DB", "#1DE23B", "#DE2621"]

    ax.hist(
        low_bea_df["loop-tof"],
        bins=bins,
        histtype="step",
        color=colors[0],
        linewidth=3.0,
        edgecolor=colors[0],
        align="mid",
        alpha=1,
        density=True,
    )
    ax.hist(
        med_bea_df["loop-tof"],
        bins=bins,
        histtype="step",
        color=colors[1],
        linewidth=3.0,
        edgecolor=colors[1],
        align="mid",
        alpha=1,
        density=True,
    )
    ax.hist(
        high_bea_df["loop-tof"],
        bins=bins,
        histtype="step",
        color=colors[2],
        linewidth=3.0,
        edgecolor=colors[2],
        align="mid",
        alpha=1,
        density=True,
    )

    fig.text(
        0.025,
        0.5,
        "Fraction of occurences",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=24,
    )

    ax.tick_params(
        which="major",
        axis="both",
        direction="out",
        right=True,
        top=True,
        bottom=True,
        length=6,
        width=1.5,
    )

    ax.legend(
        labels=[
            r"$\Delta BE_{A} = 0.3 eV$",
            r"$\Delta BE_{A} = 0.5 eV$",
            r"$\Delta BE_{A} = 0.8 eV$",
        ],
        fontsize=24,
        loc="upper right",
    )

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_xlabel(f"Loop TOF " r"$(s^{-1})$", fontsize=24)
    ax.set_xlim([-55, 55])
    ax.set_ylim([1 * 10**-5, 1 * 10**-2])
    ax.set_yscale("log")

    plt.savefig(os.path.join(FIGS_DIR, "figure3", filename), dpi=300)
    plt.show()

    return fig, ax


def main():

    set_rc_params()

    df_orig = load_csv("ml_data_op_steady.csv")

    pie_fig, pie_ax = plot_piechart(df_orig)
    hist_fig, hist_ax = plot_histogram(df_orig)
    bea_hist_fig, bea_hist_ax = plot_BEA_histogram(df_orig)

    return df_orig["loop-tof"]


if __name__ == "__main__":
    data = main()
