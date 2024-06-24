import os
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score

SEED = 17
CSVS_DIR = os.path.join("..", "..", "data", "csvs")
PKLS_DIR = os.path.join("..", "..", "data", "pkls")


def load_csv(filename, data_dir=CSVS_DIR):
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


def map_target(df, target, new_target_name, encoding):
    """
    Args:
        df (pandas.DataFrame): The dataframe to map the target classes.
        target (string): The name of the target column.
        new_target_name (string): The name of the new target column.
        encoding_map (lambda function or dict): A lambda function to map the target classes or values.
            e.g. lambda x: 1 if x > 0 else -1 if x < 0 else 0

    Returns:
        pandas.DataFrame
    """
    df[new_target_name] = df[target].map(encoding)
    return df


def load_search_results(filename, data_dir=PKLS_DIR):
    return joblib.load(os.path.join(data_dir, filename))


def get_best_model(search_results):
    return search_results.best_estimator_


def check_reg_errors(model_file, data_file, use_test_set=False):

    if use_test_set:
        print("Regression Test Results")
        print("--------------------------")
    else:
        print("Regression Training Results")
        print("------------------------------")

    df_orig = load_csv(data_file)
    df_copy = df_orig.copy()

    columns = list(df_copy)
    targets = ["loop-tof"]
    non_features = ["steady-state-condition"]
    features = [f for f in columns if f not in targets + non_features]

    condition = (df_copy["loop-tof"] <= -1e-4) | (df_copy["loop-tof"] >= 1e-4)
    df_filtered = df_copy.loc[condition]

    X_train, X_test, Y_train, Y_test = split_train_test(
        df_filtered,
        target="loop-tof",
        features=features,
        test_size=0.10,
        random_state=SEED,
        check_shape=False,
    )

    model = get_best_model(load_search_results(model_file))

    if use_test_set:
        x_true = X_test
        y_true = Y_test
    elif not use_test_set:
        x_true = X_train
        y_true = Y_train

    y_pred = model.predict(x_true)
    y = np.exp(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y))
    mae = mean_absolute_error(y_true, y)
    median_abs_err = np.median(np.abs(y_true - y))

    print(f"Model: {model_file}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"Median AE: {median_abs_err}")
    print("------------------")

    return None


def check_clf_errors(model_file, data_file, use_test_set=False):

    if use_test_set:
        print("Classification Test Results")
        print("--------------------------")
    else:
        print("Classification Training Results")
        print("------------------------------")

    df_orig = load_csv(data_file)

    columns = list(df_orig)
    targets = ["loop-tof"]
    non_features = ["steady-state-condition"]
    features = [f for f in columns if f not in targets + non_features]

    df = map_target(
        df_orig,
        target="loop-tof",
        new_target_name="class",
        encoding=lambda x: (
            1 if float("inf") > x >= 1e-4 else 2 if -float("inf") < x <= -1e-4 else 0
        ),
    )

    X_train, X_test, Y_train, Y_test = split_train_test(
        df,
        target="class",
        features=features,
        test_size=0.10,
        random_state=SEED,
        check_shape=False,
    )

    model = get_best_model(load_search_results(model_file))

    if use_test_set:
        x_true = X_test
        y_true = Y_test
    elif not use_test_set:
        x_true = X_train
        y_true = Y_train

    y_pred = model.predict(x_true)

    acc = np.mean(y_pred == y_true)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"Model: {model_file}")
    print(f"Accuracy: {acc}")
    print(f"Weighted F1: {weighted_f1}")
    print("------------------")

    return None


def main():

    warnings.filterwarnings("ignore")

    model_type = ["reg", "clf"]
    data_type = ["rc", "op"]

    for m in model_type:
        for d in data_type:
            model_file = f"{d}_steady/xgb_{m}_{d}.pkl"
            data_file = f"ml_data_{d}_steady.csv"
            if m == "reg":
                check_reg_errors(model_file, data_file, use_test_set=True)
            elif m == "clf":
                check_clf_errors(model_file, data_file, use_test_set=True)

    return None


if __name__ == "__main__":
    main()
