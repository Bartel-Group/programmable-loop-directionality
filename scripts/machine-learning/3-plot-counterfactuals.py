import os
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


def mean_perturbations_barplot(
    mean_perturbations_df, filename="mean-perturbations-barplot.png"
):
    """
    Args:
        mean_perturbations_df (dict): A dictionary of DataFrames with the mean perturbations for each class transition.

    Returns:
        None
    """
    labels = ["Positive to Negative", "Negative to Positive", "Zero to Positive"]
    colors = ["#003f5c", "#bc5090", "#ffa600"]

    fig, ax = plt.subplots()

    mean_perturbations_df.plot(kind="barh", color=colors, ax=ax)

    plt.xticks(rotation=0)
    plt.xlabel("Mean Perturbation")

    plt.ylabel("Feature")
    plt.legend(labels)

    plt.savefig(os.path.join(FIGS_DIR, "figure5", filename), dpi=300)
    plt.show()

    return None


def main():

    # set_rc_params()

    start_classes = [1, 2, 0]
    desired_classes = [2, 1, 1]

    zipped = zip(start_classes, desired_classes)

    mean_perturbations_dict = {}

    for start, desired in zipped:

        perturbations_df = load_csv(
            f"counterfactual-perturbations-{start}-to-{desired}.csv"
        )
        mean_values = perturbations_df.mean()

        mean_perturbations_dict[f"{start}_to_{desired}"] = mean_values

    mean_perturbations_df = pd.DataFrame(mean_perturbations_dict)

    mean_perturbations_barplot(mean_perturbations_df)

    return mean_perturbations_df


if __name__ == "__main__":
    df = main()
