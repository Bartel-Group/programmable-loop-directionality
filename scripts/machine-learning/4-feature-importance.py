import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

SEED = 17
CSVS_DIR = os.path.join("..", "..", "data", "csvs")
PKLS_DIR = os.path.join("..", "..", "data", "pkls")


def create_bar_plot(y_values, x_values, title="", xlabel="", ylabel=""):
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.barh(x_values, y_values)

    plt.title(title)
    plt.xlabel(xlabel, fontsize=12)

    plt.ylabel(ylabel, fontsize=12)
    plt.show()

    return None


def load_csv(filename, data_dir=CSVS_DIR):
    """
    Args:
        filename (string): The name of the file to load. (.csv)
        data_dir (string, optional): The name of the directory to load from. Defaults to CSVS_DIR.

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


def main():

    data_types = ["op", "rc"]
    model_types = ["clf", "reg"]

    for tag in data_types:
        print(f"Processing {tag} data")

        pkls_subdir = os.path.join(PKLS_DIR, tag + "_steady")

        df_orig = load_csv(filename="ml_data_" + tag + "_steady.csv")

        columns = list(df_orig)
        targets = ["loop-tof"]
        non_features = ["steady-state-condition"]
        features = [f for f in columns if f not in targets + non_features]

        for type_ in model_types:
            print(f"Processing {type_} model")

            if type_ == "clf":

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
                    os.path.join(pkls_subdir, "xgb_clf_" + tag + "-best-estm.pkl")
                )

            elif type_ == "reg":

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
                    os.path.join(pkls_subdir, "xgb_reg_" + tag + "-best-estm.pkl")
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
                    r"$\alpha_a$",
                    r"$\alpha_c$",
                    r"$\alpha_b$",
                    r"$\Delta BE_a$",
                    r"$\gamma_{c-a}$",
                    r"$\gamma_{b-a}$",
                    r"$\delta_{c-a}$",
                    r"$\delta_{b-a}$",
                    r"$\beta_b$",
                    r"$\beta_a$",
                    r"$\beta_c$",
                ]

            if tag == "rc":
                y = [
                    r"$\Delta BE_a$",
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
            )

    return None


if __name__ == "__main__":
    main()
