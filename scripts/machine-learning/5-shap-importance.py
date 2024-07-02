import os
import shap
import joblib
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

SEED = 17
CSVS_DIR = os.path.join("..", "..", "data", "csvs")
PKLS_DIR = os.path.join("..", "..", "data", "pkls")
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

    data_types = ["op"]
    model_types = ["clf"]

    for tag in data_types:

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
                    test_size=0.01,
                    random_state=SEED,
                    check_shape=False,
                )
                best_model = joblib.load(
                    os.path.join(pkls_subdir, "xgb_reg_" + tag + "-best-estm.pkl")
                )

            feature_names = [
                r"$\alpha_A$",
                r"$\alpha_B$",
                r"$\alpha_C$",
                r"$\beta_A$",
                r"$\beta_B$",
                r"$\beta_C$",
                r"$\gamma_{B-A}$",
                r"$\gamma_{C-A}$",
                r"$\delta_{B-A}$",
                r"$\delta_{C-A}$",
                r"$\Delta BE_A$",
            ]

            explainer = shap.Explainer(best_model)
            shap_values = explainer(X_test)

            if type_ == "clf":

                fig = plt.figure(figsize=(12, 12))

                ax = plt.subplot(2, 2, 1)
                ax = pkl.load(
                    open(
                        os.path.join(pkls_subdir, "clf-op-feature-importance-ax.pkl"),
                        "rb",
                    )
                )

                ax = plt.subplot(2, 2, 2)
                ax = shap.summary_plot(
                    shap_values[:, :, 0],
                    feature_names=feature_names,
                    show=False,
                )
                ax = plt.title("Class 0: Zero loop TOF")

                ax = plt.subplot(2, 2, 3)
                ax = shap.summary_plot(
                    shap_values[:, :, 1],
                    feature_names=feature_names,
                    show=False,
                )
                ax = plt.title("Class 1: Positive loop TOF")

                ax = plt.subplot(2, 2, 4)
                ax = shap.summary_plot(
                    shap_values[:, :, 2],
                    feature_names=feature_names,
                    show=False,
                )
                ax = plt.title("Class 2: Negative loop TOF")
                ax = plt.savefig(
                    os.path.join(FIGS_DIR, "figure4", "class-2-shap-op.png"), dpi=600
                )
                ax = plt.show()

            elif type_ == "reg":

                shap.summary_plot(shap_values, feature_names=feature_names, show=False)
                plt.savefig(
                    os.path.join(FIGS_DIR, "figure7", "reg-shap-op.png"), dpi=300
                )
                plt.show()

    return None


if __name__ == "__main__":
    main()
