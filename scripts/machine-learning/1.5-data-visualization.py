import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

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


def plot_piechart(df):
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
        autopct="%1.1f%%",
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
    plt.show()

    return fig, ax


def plot_histogram(df):
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
    ax[0].legend(labels=["Positive"], fontsize=20, loc="upper right", frameon=False)

    ax[1].hist(
        -neg_df["loop-tof"], bins, color="#ffa600", linewidth=1.0, edgecolor="black"
    )
    ax[1].legend(labels=["Negative"], fontsize=20, loc="lower right", frameon=False)

    for k in range(2):
        ax[k].set_xlim([5 * 1e-5, 1e2])
        ax[k].set_xscale("log")
        ax[k].set_yscale("log")

    fig.text(
        0.025,
        0.5,
        "Frequency of Output",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=24,
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
    ax[1].set_xlabel(f"Loop TOF " r"$(\frac{1}{s})$", fontsize=24)
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

    plt.show()

    return fig, ax


def plot_BEA_histogram(df):

    pos_df, neg_df, zero_df = split_TOFs_by_sign(df)
    pos_low_bea_df, pos_med_bea_df, pos_high_bea_df = split_TOFs_by_BEA(pos_df)
    neg_low_bea_df, neg_med_bea_df, neg_high_bea_df = split_TOFs_by_BEA(neg_df)

    pos_bea_df = pd.DataFrame(
        data={
            "Positive, ΔBEa = 0.3 eV": pos_low_bea_df["loop-tof"],
            "Positive, ΔBEa = 0.5 eV": pos_med_bea_df["loop-tof"],
            "Positive, ΔBEa = 0.8 eV": pos_high_bea_df["loop-tof"],
        }
    )
    neg_bea_df = pd.DataFrame(
        data={
            "Negative, ΔBEa = 0.3 eV": -neg_low_bea_df["loop-tof"],
            "Negative, ΔBEa = 0.5 eV": -neg_med_bea_df["loop-tof"],
            "Negative, ΔBEa = 0.8 eV": -neg_high_bea_df["loop-tof"],
        }
    )

    bins = np.logspace(np.log10(1e-4), np.log10(1e2), 14)

    fig, ax = plt.subplots(2, 1, figsize=(14, 10), dpi=300, gridspec_kw={"hspace": 0.0})
    colors = ["#003f5c", "#955196", "#ff6e54", "#444e86", "#dd5182", "#ffa600"]

    # plot each subset
    # Plot the stacked histograms for positive subsets on ax[1]
    ax[0].hist(
        [pos_bea_df[col] for col in pos_bea_df.columns],
        bins,
        histtype="bar",
        color=colors[:3],
        linewidth=1.0,
        edgecolor="black",
    )
    ax[0].legend(
        labels=[
            "Positive, ΔBEa = 0.3 eV",
            "Positive, ΔBEa = 0.5 eV",
            "Positive, ΔBEa = 0.8 eV",
        ],
        fontsize=12,
        loc="upper right",
    )

    # Plot the stacked histograms for negative subsets on ax[2]
    ax[1].hist(
        [neg_bea_df[col] for col in neg_bea_df.columns],
        bins,
        histtype="bar",
        color=colors[3:],
        linewidth=1.0,
        edgecolor="black",
    )
    ax[1].legend(
        labels=[
            "Negative, ΔBEa = 0.3 eV",
            "Negative, ΔBEa = 0.5 eV",
            "Negative, ΔBEa = 0.8 eV",
        ],
        fontsize=12,
        loc="lower right",
    )

    for k in range(2):
        ax[k].set_xlim([5 * 1e-5, 2 * 1e2])
        ax[k].set_xscale("log")
        ax[k].set_yscale("log")

    fig.text(
        0.025,
        0.5,
        "Frequency of Output",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=24,
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
    ax[1].set_xlabel(f"Loop TOF " r"$(\frac{1}{s})$", fontsize=24)
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

    plt.show()

    return fig, ax


def main():

    set_rc_params()

    df_orig = load_csv("ml_data_op_steady.csv")
    pie_fig, pie_ax = plot_piechart(df_orig)
    hist_fig, hist_ax = plot_histogram(df_orig)
    bea_hist_fig, bea_hist_ax = plot_BEA_histogram(df_orig)

    return None


if __name__ == "__main__":
    main()
