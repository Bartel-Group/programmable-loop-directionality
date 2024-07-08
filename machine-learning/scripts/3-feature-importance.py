import os
import joblib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

SEED = 17
TRAINING_DATA_DIR = os.path.join("..", "training-data")
MODELS_DIR = os.path.join("..", "grid-search-results")
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
        "font.size": 12,
        "figure.dpi": 300,
        "font.family": "arial",
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "axes.edgecolor": "black",
        "figure.figsize": [10, 10],
    }
    for p in params:
        mpl.rcParams[p] = params[p]
    return params


def load_csv(filename, data_dir=TRAINING_DATA_DIR):
    """
    Args:
        filename (string): The name of the file to load. (.csv)
        data_dir (string, optional): The name of the directory to load from. Defaults to TRAINING_DATA_DIR.

    Returns:
        pandas.DataFrame
    """
    return pd.read_csv(os.path.join(data_dir, filename))


def map_target(df, target, new_target_name, encoding):
    """
    Args:
        df (pandas.DataFrame): The dataframe to map the target classes.
        target (string): The name of the target column.
        encoding (lambda function): A lambda function to map the target classes.
            e.g. lambda x: 1 if x > 0 else -1 if x < 0 else 0

    Returns:
        pandas.DataFrame
    """
    df[new_target_name] = df[target].map(encoding)
    return df


def split_train_test(
    df,
    target,
    features,
    test_size=0.10,
    random_state=SEED,
    check_shape=True,
):
    """
    Args:
        df (pandas.DataFrame): The dataframe to split.
        target (string): The name of the target column.
        features (list, optional): The list of feature column names to use.
        test_size (float, optional): The proportion of the data to include in the test split. Defaults to 0.10.
        random_state (int, optional): Controls the shuffling applied to the data before applying the split. Defaults to the global seed.
        check_shape (bool, optional): If True, will check the shape of the training and testing data. Defaults to True.

    Returns:
        (lists) : X_train, X_test, y_train, y_test
    """
    assert len(features) > 0, "No feature columns provided"

    Y = df[target].values.reshape(-1, 1)
    X = df[features].values

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    if check_shape:
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {Y_train.shape}")
        print(f"y_test shape: {Y_test.shape}")

        if X_train.shape[0] != Y_train.shape[0]:
            raise ValueError("X_train and Y_train have different number of samples")

    return X_train, X_test, Y_train, Y_test


def create_bar_plot(
    y_values, x_values, title="", xlabel="", ylabel="", filename="barplot.png"
):
    fig, ax = plt.subplots(figsize=(6, 4))

    ax = plt.barh(x_values, y_values, color="#003f5c", edgecolor="black", linewidth=0.5)

    ax = plt.title(title)
    ax = plt.xlabel(xlabel, fontsize=12)
    ax = plt.ylabel(ylabel, fontsize=12)

    plt.savefig(filename, dpi=300)
    ax = plt.show()

    return None


def main():

    set_rc_params()

    data_types = ["op"]
    model_types = ["clf", "reg"]

    for tag in data_types:
        print(f"Processing {tag} data")

        df_orig = load_csv(filename="ml_data_" + tag + "_steady.csv")

        columns = list(df_orig)
        targets = ["loop-tof"]
        non_features = ["steady-state-condition"]
        features = [f for f in columns if f not in targets + non_features]

        for type_ in model_types:
            print(f"Processing {type_} model")

            if type_ == "clf":

                filename = os.path.join(
                    FIGS_DIR, "figure4", f"{tag}-{type_}-feature-importance.png"
                )

                df = map_target(
                    df_orig,
                    target="loop-tof",
                    new_target_name="direction",
                    encoding=lambda x: (
                        1
                        if float("inf") > x >= 1e-4
                        else 2 if -float("inf") < x <= -1e-4 else 0
                    ),
                )

                X_train, X_test, Y_train, Y_test = split_train_test(
                    df,
                    target="direction",
                    features=features,
                    test_size=0.10,
                    random_state=SEED,
                    check_shape=False,
                )

                best_model = joblib.load(
                    os.path.join(MODELS_DIR, "xgb_clf_" + tag + "-best-estm.pkl")
                )

            elif type_ == "reg":

                filename = os.path.join(
                    FIGS_DIR, "figure7", f"{tag}-{type_}-feature-importance.png"
                )

                df_copy = df_orig.copy()
                condition = (df_copy["loop-tof"] <= -1e-4) | (
                    df_copy["loop-tof"] >= 1e-4
                )
                df_filtered = df_copy.loc[condition]

                df = map_target(
                    df_filtered,
                    target="loop-tof",
                    new_target_name="log-loop-tof",
                    encoding=lambda x: np.log(np.abs(x)),
                )

                X_train, X_test, Y_train, Y_test = split_train_test(
                    df,
                    target="log-loop-tof",
                    features=features,
                    test_size=0.10,
                    random_state=SEED,
                    check_shape=False,
                )
                best_model = joblib.load(
                    os.path.join(MODELS_DIR, "xgb_reg_" + tag + "-best-estm.pkl")
                )

            results = permutation_importance(
                best_model, X_test, Y_test, n_repeats=10, random_state=17, n_jobs=-1
            )

            feat_importances = {
                features[i]: results.importances_mean[i] for i in range(len(features))
            }
            sorted_imp = {
                k: v
                for k, v in sorted(
                    feat_importances.items(), key=lambda item: item[1], reverse=False
                )
            }

            if tag == "op":
                y = [
                    r"$\alpha_A$",
                    r"$\alpha_C$",
                    r"$\alpha_B$",
                    r"$\Delta BE_A$",
                    r"$\gamma_{C-A}$",
                    r"$\gamma_{B-A}$",
                    r"$\delta_{C-A}$",
                    r"$\delta_{B-A}$",
                    r"$\beta_B$",
                    r"$\beta_A$",
                    r"$\beta_C$",
                ]

            if tag == "rc":
                y = [
                    r"$\Delta BE_A$",
                    r"$k_{-" + str(1) + r"," + str(2) + r"}$",
                    r"$k_{-" + str(3) + r"," + str(1) + r"}$",
                    r"$k_{" + str(1) + r"," + str(1) + r"}$",
                    r"$k_{" + str(3) + r"," + str(2) + r"}$",
                    r"$k_{-" + str(2) + r"," + str(1) + r"}$",
                    r"$k_{-" + str(1) + r"," + str(1) + r"}$",
                    r"$k_{-" + str(3) + r"," + str(2) + r"}$",
                    r"$k_{-" + str(2) + r"," + str(2) + r"}$",
                    r"$k_{" + str(3) + r"," + str(1) + r"}$",
                    r"$k_{" + str(2) + r"," + str(1) + r"}$",
                    r"$k_{" + str(2) + r"," + str(2) + r"}$",
                    r"$k_{" + str(1) + r"," + str(2) + r"}$",
                ]

            create_bar_plot(
                list(sorted_imp.values()),
                y,
                title="",
                xlabel="Feature Importance",
                filename=filename,
            )

    return None


if __name__ == "__main__":
    main()
