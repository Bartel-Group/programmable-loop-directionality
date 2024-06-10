import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score

SEED = 17
DATA_DIR = "/Users/noord014/MyDrive/phd/research/dmc-lab/bartel-group/programmable-catalysis/dev/data/pkls"
FIG_DIR = "/Users/noord014/MyDrive/phd/research/dmc-lab/bartel-group/programmable-catalysis/dev/figures"


def load_data(filename, data_dir=DATA_DIR):
    return pd.read_csv(os.path.join(data_dir, filename))


def split_train_test(
    df,
    target,
    features,
    test_size=0.10,
    random_state=SEED,
    check_shape=True,
):
    assert len(features) > 0, "No feature columns provided"

    y = df[target].values.reshape(-1, 1)
    X = df[features].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if check_shape:
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train have different number of samples")

    return X_train, X_test, y_train, y_test


def map_target_classes(df, target, encoding):
    """
    Args:
        df (pandas.DataFrame): The dataframe to map the target classes.
        target (string): The name of the target column.
        encoding_map (lambda function or dict): A lambda function to map the target classes or a dictionary mapping the target classes to the new encoding.
            e.g. lambda x: 1 if x > 0 else -1 if x < 0 else 0 or {"Al": 0, "O": 1, "Au": -1}

    Returns:
        pandas.DataFrame
    """
    df["class"] = df[target].map(encoding)
    return df


def load_search_results(filename, data_dir=DATA_DIR):
    return joblib.load(os.path.join(data_dir, filename))


def get_best_model(search_results):
    return search_results.best_estimator_


def check_reg_errors(use_test_set=False):

    filenames = [
        "rc_steady/xgb_reg_rc-random-cv-no-zeros.pkl",
        "op_steady/xgb_reg_op-random-cv-no-zeros.pkl",
    ]

    original_data = [
        "240313_rc_steady.csv",
        "240313_op_steady.csv",
    ]

    zipped = zip(filenames, original_data)

    if use_test_set:
        print("Regression Test Results")
        print("--------------------------")
    else:
        print("Regression Training Results")
        print("------------------------------")

    for f, d in zipped:
        df_orig = load_data(os.path.join(DATA_DIR, "..", "csvs", d))

        columns = list(df_orig)
        targets = ["loop-tof"]
        non_features = ["steady-state-condition"]
        features = [f for f in columns if f not in targets + non_features]

        df_orig = df_orig[df_orig["loop-tof"] != 0]

        X_train, X_test, y_train, y_test = split_train_test(
            df_orig,
            target="loop-tof",
            features=features,
            test_size=0.10,
            random_state=SEED,
            check_shape=False,
        )

        model = get_best_model(load_search_results(f))

        if use_test_set:
            x_true = X_test
            y_true = y_test
        elif not use_test_set:
            x_true = X_train
            y_true = y_train

        y_pred = model.predict(x_true)
        y = np.exp(y_pred)

        rmse = np.sqrt(mean_squared_error(y_true, y))
        mae = mean_absolute_error(y_true, y)
        median_abs_err = np.median(np.abs(y_true - y))

        print(f"Model: {f}")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"Median AE: {median_abs_err}")
        print("------------------")

    return None


def check_clf_errors(use_test_set=False):

    filenames = [
        "rc_steady/xgb_clf_rc-random-cv.pkl",
        "op_steady/xgb_clf_op-random-cv.pkl",
    ]

    original_data = [
        "240313_rc_steady.csv",
        "240313_op_steady.csv",
    ]

    zipped = zip(filenames, original_data)

    if use_test_set:
        print("Classification Test Results")
        print("--------------------------")
    else:
        print("Classification Training Results")
        print("------------------------------")

    for f, d in zipped:
        df_orig = load_data(os.path.join(DATA_DIR, "..", "csvs", d))

        columns = list(df_orig)
        targets = ["loop-tof"]
        non_features = ["steady-state-condition"]
        features = [f for f in columns if f not in targets + non_features]

        df = map_target_classes(
            df_orig,
            target="loop-tof",
            encoding=lambda x: 1 if x > 0 else 2 if x < 0 else 0,
        )

        X_train, X_test, y_train, y_test = split_train_test(
            df,
            target="class",
            features=features,
            test_size=0.10,
            random_state=SEED,
            check_shape=False,
        )

        model = get_best_model(load_search_results(f))

        if use_test_set:
            x_true = X_test
            y_true = y_test
        elif not use_test_set:
            x_true = X_train
            y_true = y_train

        y_pred = model.predict(x_true)

        acc = np.mean(y_pred == y_true)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted")
        print(f"Model: {f}")
        print(f"Accuracy: {acc}")
        print(f"Weighted F1: {weighted_f1}")
        print("------------------")

    return None


def main():
    check_reg_errors(use_test_set=True)
    check_clf_errors(use_test_set=True)
    return None


if __name__ == "__main__":
    main()
