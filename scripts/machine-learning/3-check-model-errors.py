import os
import joblib
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score

SEED = 17
CSVS_DIR = os.path.join("..", "..", "data", "csvs")
PKLS_DIR = os.path.join("..", "..", "data", "pkls")
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
        "xtick.top": False,
        "ytick.right": True,
        "axes.edgecolor": "black",
        "figure.figsize": [6, 4],
    }
    for p in params:
        mpl.rcParams[p] = params[p]
    return params


def load_csv(filename, data_dir=CSVS_DIR):
    return pd.read_csv(os.path.join(data_dir, filename))


def load_best_model(filename, data_dir=PKLS_DIR):
    return joblib.load(os.path.join(data_dir, filename))


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


def plot_hexbin(true_vals, predicted_vals, params, filename="parity-plot.png"):
    sns.set_theme(style="ticks", rc=params)
    grid = sns.jointplot(
        x=predicted_vals,
        y=true_vals,
        kind="hex",
        cmap="plasma",
        mincnt=1,
        bins="log",
        marginal_ticks=False,
        ratio=5,
        space=0,
        xscale="log",
        yscale="log",
        marginal_kws=dict(bins=46, fill=True, color="#0C0881", kde=True),
    )
    grid.ax_joint.plot(
        [1 * 10**-5, 1 * 10**3],
        [1 * 10**-5, 1 * 10**3],
        "--",
        color="red",
        linewidth=3,
    )

    grid.ax_joint.set_xlim(4 * 10**-5, 3 * 10**2)
    grid.ax_joint.set_ylim(4 * 10**-5, 3 * 10**2)
    grid.ax_joint.set_xticks(np.logspace(-4, 2, num=7))
    grid.ax_joint.set_yticks(np.logspace(-4, 2, num=7))
    grid.ax_joint.set_xlabel(r"$|TOF_{pred}|$" + " " + r"$(s^{-1})$", size=24)
    grid.ax_joint.set_ylabel(r"$|TOF_{true}|$" + " " + r"$(s^{-1})$", size=24)

    grid.figure.set_figwidth(12)
    grid.figure.set_figheight(9)

    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    cbar_ax = grid.figure.add_axes([0.80, 0.20, 0.02, 0.5])  # x, y, width, height
    plt.colorbar(cax=cbar_ax).set_label(label=r"Count", size=26)

    plt.savefig(os.path.join(FIGS_DIR, "figure6", filename), dpi=300)
    plt.show()

    return None


def check_reg_errors(
    model_file, data_file, use_test_set=False, check_baseline=False, parity_plot=False
):

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

    model = load_best_model(model_file)

    if use_test_set:
        x_true = X_test
        y_true = Y_test
    elif not use_test_set:
        x_true = X_train
        y_true = Y_train

    y_pred = model.predict(x_true)
    y = np.exp(y_pred)

    y_true = np.abs(np.array(y_true.flatten()))

    rmse = np.sqrt(mean_squared_error(y_true, y))
    mae = mean_absolute_error(y_true, y)
    median_abs_err = np.median(np.abs(y_true - y))
    seventyfifth_percentile = np.percentile(np.abs(y_true - y), 75)

    print(f"Model: {model_file}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"Median AE: {median_abs_err}")
    print(f"75th Percentile AE: {seventyfifth_percentile}")
    print("------------------")

    if check_baseline:
        mean_baseline = np.mean(y_true)
        mean_baseline_pred = np.full(len(y_true), mean_baseline)

        med_baseline = np.median(y_true)
        print(f"Median: {med_baseline}")
        med_baseline_pred = np.full(len(y_true), med_baseline)

        rmse_baseline = np.sqrt(mean_squared_error(y_true, mean_baseline_pred))
        mae_baseline = mean_absolute_error(y_true, mean_baseline_pred)
        median_abs_err_baseline = np.median(np.abs(y_true - med_baseline_pred))

        print(f"Baseline RMSE: {rmse_baseline}")
        print(f"Baseline MAE: {mae_baseline}")
        print(f"Baseline Median AE: {median_abs_err_baseline}")
        print("------------------")
        print("\n")

    else:
        print("\n")

    if parity_plot:
        plot_hexbin(
            y_true,
            y,
            params=set_rc_params(),
            filename=model_file.split("_")[0] + "-parity-plot.png",
        )

    return None


def check_clf_errors(model_file, data_file, use_test_set=False, check_baseline=False):

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

    model = load_best_model(model_file)

    if use_test_set:
        x_true = X_test
        y_true = Y_test
    elif not use_test_set:
        x_true = X_train
        y_true = Y_train

    y_pred = model.predict(x_true)
    baseline_zero_pred = np.zeros(len(y_true))
    baseline_positive_pred = np.ones(len(y_true))
    baseline_negative_pred = np.full(len(y_true), 2)

    acc = np.mean([y_true[i] == y_pred[i] for i in range(len(y_pred))])
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"Model: {model_file}")
    print(f"Accuracy: {acc}")
    print(f"Weighted F1: {weighted_f1}")
    print("------------------")

    if check_baseline:
        acc_zero = np.mean(
            [y_true[i] == baseline_zero_pred[i] for i in range(len(baseline_zero_pred))]
        )
        acc_pos = np.mean(
            [
                y_true[i] == baseline_positive_pred[i]
                for i in range(len(baseline_positive_pred))
            ]
        )
        acc_neg = np.mean(
            [
                y_true[i] == baseline_negative_pred[i]
                for i in range(len(baseline_negative_pred))
            ]
        )

        print(f"Baseline Zero Accuracy: {acc_zero}")
        print(f"Baseline Positive Accuracy: {acc_pos}")
        print(f"Baseline Negative Accuracy: {acc_neg}")
        print("------------------")

        weighted_f1_zero = f1_score(y_true, baseline_zero_pred, average="weighted")
        weighted_f1_pos = f1_score(y_true, baseline_positive_pred, average="weighted")
        weighted_f1_neg = f1_score(y_true, baseline_negative_pred, average="weighted")

        print(f"Baseline Zero Weighted F1: {weighted_f1_zero}")
        print(f"Baseline Positive Weighted F1: {weighted_f1_pos}")
        print(f"Baseline Negative Weighted F1: {weighted_f1_neg}")
        print("------------------")
        print("\n")

    else:
        print("\n")

    return None


def main():

    # warnings.filterwarnings("ignore")

    model_type = ["reg", "clf"]
    data_type = ["rc", "op"]

    for m in model_type:
        for d in data_type:
            model_file = f"{d}_steady/xgb_{m}_{d}-best-estm.pkl"
            data_file = f"ml_data_{d}_steady.csv"
            if m == "reg":
                check_reg_errors(
                    model_file,
                    data_file,
                    use_test_set=True,
                    check_baseline=True,
                    parity_plot=True,
                )
            elif m == "clf":
                check_clf_errors(
                    model_file, data_file, use_test_set=True, check_baseline=True
                )

    return None


if __name__ == "__main__":
    main()
