import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

COUNTERFACTUAL_RESULTS_DIR = os.path.join("..", "counterfactual-results")
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
        "figure.figsize": [12, 8],
    }
    for p in params:
        mpl.rcParams[p] = params[p]
    return params


def load_csv(filename, data_dir=COUNTERFACTUAL_RESULTS_DIR):
    """
    Args:
        filename (string): The name of the file to load. (.csv)
        data_dir (string, optional): The name of the directory to load from. Defaults to CSVS_DIR.

    Returns:
        pandas.DataFrame
    """
    return pd.read_csv(os.path.join(data_dir, filename))


def mean_perturbations_barplot(
    mean_perturbations_df, filename="mean-perturbations-barplot.png"
):
    """
    Args:
        mean_perturbations_df (dict): A dictionary of DataFrames with the mean perturbations for each class transition.
        filename (string, optional): The name of the file to save the figure. Defaults to "mean-perturbations-barplot.png".

    Returns:
        None
    """
    labels = ["Positive to Negative", "Negative to Positive", "Zero to Positive"]
    colors = ["#003f5c", "#bc5090", "#ffa600"]

    fig, ax = plt.subplots()

    mean_perturbations_df.plot(kind="barh", color=colors, ax=ax, edgecolor="black")

    ax.tick_params(
        which="minor",
        axis="x",
        direction="out",
        right=False,
        top=False,
        length=4,
        width=1.5,
    )
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.xlabel("Mean Perturbation (Normalized)")
    plt.legend(labels, loc="upper right")

    plt.savefig(os.path.join(FIGS_DIR, "figure5", filename), dpi=300)
    plt.show()

    return None


def main():

    set_rc_params()

    base_model = ["xgb", "rf"]
    start_classes = [1, 2, 0]
    desired_classes = [2, 1, 1]

    mean_perturbations_dict = {}

    for b in base_model:
        mean_perturbations_dict[b] = {}
        zipped = zip(start_classes, desired_classes)

        for start, desired in zipped:

            perturbations_df = load_csv(
                f"{b}-counterfactual-perturbations-{start}-to-{desired}.csv"
            )

            perturbations_df = perturbations_df.replace(0.0, np.nan)

            stacked_alpha_values = (
                perturbations_df[["alpha-a", "alpha-b", "alpha-c"]]
                .stack(dropna=True)
                .reset_index(drop=True)
            )
            stacked_beta_values = (
                perturbations_df[["beta-a", "beta-b", "beta-c"]]
                .stack(dropna=True)
                .reset_index(drop=True)
            )
            stacked_gamma_values = (
                perturbations_df[["gamma-b-a", "gamma-c-a"]]
                .stack(dropna=True)
                .reset_index(drop=True)
            )
            stacked_delta_values = (
                perturbations_df[["delta-b-a", "delta-c-a"]]
                .stack(dropna=True)
                .reset_index(drop=True)
            )

            alpha_mean_values = stacked_alpha_values.mean()
            beta_mean_values = stacked_beta_values.mean()
            gamma_mean_values = stacked_gamma_values.mean()
            delta_mean_values = stacked_delta_values.mean()
            bea_mean_values = perturbations_df["change-in-bea"].dropna().mean()

            alpha_range = 0.7
            beta_range = 0.6
            gamma_range = 1.2
            delta_range = 1.0
            bea_range = 0.5

            normalized_alpha_mean_values = alpha_mean_values / alpha_range
            normalized_beta_mean_values = beta_mean_values / beta_range
            normalized_gamma_mean_values = gamma_mean_values / gamma_range
            normalized_delta_mean_values = delta_mean_values / delta_range
            normalized_bea_mean_values = bea_mean_values / bea_range

            mean_perturbations_dict[b][f"{start}_to_{desired}"] = {
                r"$\alpha_i$": normalized_alpha_mean_values,
                r"$\beta_i$": normalized_beta_mean_values,
                r"$\gamma_{i-j}$": normalized_gamma_mean_values,
                r"$\delta_{i-j}$": normalized_delta_mean_values,
                r"$\Delta BE_A$": normalized_bea_mean_values,
            }

        mean_perturbations_df = pd.DataFrame(mean_perturbations_dict[b])
        print(mean_perturbations_df.head())

        mean_perturbations_barplot(
            mean_perturbations_df, filename=f"{b}-mean-perturbations-barplot.png"
        )

    return None


if __name__ == "__main__":
    main()
