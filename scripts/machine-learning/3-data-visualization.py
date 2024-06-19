import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSVS_DIR = os.path.join("..", "..", "data", "csvs")
FIGS_DIR = os.path.join("..", "..", "figures")


def load_csv(filename, data_dir=CSVS_DIR):
    """
    Args:
        filename (string): The name of the file to load. (.csv)
        data_dir (string, optional): The name of the directory to load from. Defaults to CSVS_DIR.

    Returns:
        pandas.DataFrame
    """
    return pd.read_csv(os.path.join(data_dir, filename))


def split_TOFs(df):
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


def plot_piechart(df):
    """
    Args:
        df (pandas.DataFrame): The DataFrame to plot
    """
    pos_df, neg_df, zero_df = split_TOFs(df)

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
    plt.savefig(os.path.join(FIGS_DIR, "TOF_distribution.png"))
    plt.show()

    return fig, ax


def plot_histogram(df):
    """
    Args:
        df (pandas.DataFrame): The DataFrame to plot
    """

    pos_df, neg_df, zero_df = split_TOFs(df)
    bins = np.logspace(np.log10(1e-4), np.log10(1e2), 100)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), dpi=300, gridspec_kw={"hspace": 0.0})

    ax[0].hist(
        pos_df["loop-tof"], bins, color="#2f4b7c", linewidth=1.0, edgecolor="black"
    )
    ax[0].legend(labels=["Positive"], fontsize=16, loc="upper right", frameon=False)

    ax[1].hist(
        -neg_df["loop-tof"], bins, color="#ffa600", linewidth=1.0, edgecolor="black"
    )
    ax[1].legend(labels=["Negative"], fontsize=16, loc="lower right", frameon=False)

    for k in range(2):
        ax[k].set_xlim([5 * 1e-5, 1e2])
        ax[k].set_xscale("log")
        ax[k].set_yscale("log")

        ax[k].tick_params(
            which="major",
            axis="both",
            direction="out",
            right=True,
            length=6,
            width=1.5,
        )

    fig.text(
        0.05,
        0.5,
        "Frequency of Output",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=16,
    )
    ax[0].set_xticks([])
    ax[0].set_yticks([1, 10**1, 10**2, 10**3])

    ax[1].invert_yaxis()
    ax[1].set_yticks([1, 10**1, 10**2, 10**3])
    ax[1].set_xlabel(f"Loop TOF " r"$(s^{-1})$", fontsize=16)

    # plt.tight_layout()
    # plt.savefig("LoopTOF_histogram.png")

    return fig, ax


def main():

    df_orig = load_csv("ml_data_op_steady.csv")
    plot_piechart(df_orig)
    plot_histogram(df_orig)

    return None


if __name__ == "__main__":
    main()
