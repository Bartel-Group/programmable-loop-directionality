import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

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


def load_search_results(filename, data_dir=DATA_DIR):
    return joblib.load(os.path.join(data_dir, filename))


def get_cv_results(search_results):
    return search_results.cv_results_


def get_best_model(search_results):
    return search_results.best_estimator_


def get_best_model_idx(cv_results):
    cv_ranking = cv_results["rank_test_score"]
    best_model_idx = np.where(cv_ranking == 1)[0][0]
    return best_model_idx


def get_best_train_val_scores(cv_results, best_model_idx):
    train_scores = []
    val_scores = []

    for i in range(0, 9):
        train_scores.append(cv_results[f"split{i}_train_score"][best_model_idx])
        val_scores.append(cv_results[f"split{i}_test_score"][best_model_idx])

    return train_scores, val_scores


def generate_data_dict(filenames):
    data = {}
    for filename in filenames:
        parent_dir, file = filename.split("/")
        _, model_type, params = file.split("_")
        if parent_dir not in data:
            data[parent_dir] = {}
        search_results = load_search_results(filename)
        cv_results = get_cv_results(search_results)
        best_model_idx = get_best_model_idx(cv_results)
        train_scores, val_scores = get_best_train_val_scores(cv_results, best_model_idx)

        data[parent_dir][model_type] = {
            "params": params,
            "train_scores": train_scores,
            "val_scores": val_scores,
        }

    return data


def plot_cv_scores(data):
    fig, ax = plt.subplots(1, 2, figsize=(12, 12))
    for parent_dir, models in data.items():
        for i, (model_type, model_data) in enumerate(models.items()):
            ax[i].plot(model_data["train_scores"], label=f"{parent_dir} - Train")
            ax[i].plot(model_data["val_scores"], label=f"{parent_dir} - Validation")
            ax[i].set_title(f"{model_type} - Train vs Validation Scores")
            ax[i].set_xlabel("CV Fold")
            ax[i].set_ylabel("Score")
            ax[i].legend()
            ax[i].grid(True)

    return fig, ax


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
            x_case = X_test
            y_case = y_test
        elif not use_test_set:
            x_case = X_train
            y_case = y_train

        y_pred = model.predict(x_case)
        y = np.exp(y_pred)

        rmse = np.sqrt(mean_squared_error(y_case, y))
        mae = mean_absolute_error(y_case, y)
        median_abs_err = np.median(np.abs(y_case - y))

        print(f"Model: {f}")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"Median AE: {median_abs_err}")
        print("------------------")

    return None


def main():
    check_reg_errors(use_test_set=True)
    return None


if __name__ == "__main__":
    main()
