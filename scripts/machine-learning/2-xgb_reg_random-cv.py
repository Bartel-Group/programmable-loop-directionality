import os
import joblib
import numpy as np
import pandas as pd
import scipy.stats as st

from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

SEED = 17
DATA_DIR = "/home/cbartel/noord014/projects/programmable-catalysis/data/rc_steady/regression/csvs"
SAVE_DIR = "/home/cbartel/noord014/projects/programmable-catalysis/data/rc_steady/regression/pkls"


def load_data(filename, data_dir=DATA_DIR):
    """
    Args:
        filename (string): The name of the file to load. (.csv)
        data_dir (string, optional): The name of the directory to load from. Defaults to DATA_DIR.

    Returns:
        pandas.DataFrame
    """
    return pd.read_csv(os.path.join(data_dir, filename))


def map_target_values(df, target, encoding, target_name="target"):
    """
    Args:
        df (pandas.DataFrame): The dataframe to map the target classes.
        target (string): The name of the target column to be mapped.
        target_name (string, optional): The name of the new target column. Defaults to "target".
        encoding_map (lambda function or dict): A lambda function to map the original target to other values
            e.g. lambda x: np.log(x)

    Returns:
        pandas.DataFrame
    """
    df[target_name] = df[target].map(encoding)
    return df


def split_train_test(
    df,
    target,
    features,
    test_size=0.10,
    scale_features=False,
    scale_target=False,
    scaler=StandardScaler(),
    random_state=SEED,
    check_shape=True,
):
    """
    Args:
        df_orig (pandas.DataFrame): The dataframe to split.
        target (string): The name of the target column.
        features (list, optional): The list of feature column names to use.
        test_size (float, optional): The proportion of the data to include in the test split. Defaults to 0.10.
        random_state (int, optional): Controls the shuffling applied to the data before applying the split. Defaults to the global seed.
        check_shape (bool, optional): If True, will check the shape of the training and testing data. Defaults to True.

    Returns:
        (lists) : X_train, X_test, y_train, y_test
    """
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

    if scale_features:
        fitted_scaler = scaler.fit(X_train)
        X_train_scaled = fitted_scaler.transform(X_train)
        X_test_scaled = fitted_scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test
    elif scale_target:
        fitted_scaler = scaler.fit(y_train)
        y_train_scaled = fitted_scaler.transform(y_train)
        y_test_scaled = fitted_scaler.transform(y_test)
        return X_train, X_test, y_train_scaled, y_test_scaled
    else:
        return X_train, X_test, y_train, y_test


def search_random_params(
    X_train,
    y_train,
    model,
    param_dists,
    cv=5,
    n_iters=100,
    n_jobs=1,
    verbose=2,
    scoring="neg_mean_absolute_error",
    keep_trian_score=False,
    filename="xgb_reg_rc-random-cv.pkl",
    save_dir=SAVE_DIR,
    save_best=True,
    best_name="xgb_reg_rc-best-estm.pkl",
    remake=False,
):
    """
    Args:
        X_train (array-like): The training input samples.
        y_train (array-like): The target values.
        model (sklearn.base.BaseEstimator): The model to fit.
        param_dists (dict): Dictionary with parameters names (string) as keys and distributions or lists of parameters to try.
        cv (int, cross-validation generator or an iterable, optional): Determines the cross-validation splitting strategy. Defaults to 5.
        n_iters (int, optional): Number of parameter settings that are sampled. Defaults to 100.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to 4. (n_jobs=n_cores if on MSI)
        verbose (int, optional): Controls the verbosity. Defaults to 2.
        scoring (string, callable, list/tuple, dict or None, optional): A single string (see The scoring parameter: defining model evaluation rules) or a callable (see Defining your scoring strategy from metric functions) to evaluate the predictions on the test set. Defaults to "f1_weighted".
        keep_trian_score (bool, optional): If False, the cv_results_ attribute will not include training scores. Defaults to False for memory efficiency.
        filename (string, optional): The name of the file to save the search. Defaults to "xgb_reg_random-cv.pkl".
        save_dir (string, optional): The name of the directory to save the search. Defaults to DATA_DIR.
        save_best (bool, optional): If True, will save the best estimator. Defaults to True.
        best_name (string, optional): The name of the file to save the best estimator. Defaults to "xgb_reg_best-estm.pkl".
        remake (bool, optional): If True, will remake the search. Defaults to False.

    Returns:
        sklearn.model_selection._search.GridSearchCV
    """

    if not remake and os.path.exists(os.path.join(save_dir, filename)):
        return joblib.load(os.path.join(save_dir, filename))

    gs = RandomizedSearchCV(
        model,
        param_dists,
        cv=cv,
        n_iter=n_iters,
        n_jobs=n_jobs,
        verbose=verbose,
        scoring=scoring,
        random_state=SEED,
        return_train_score=keep_trian_score,
    )
    gs.fit(X_train, y_train)

    if save_best:
        best_estimator = gs.best_estimator_
        joblib.dump(best_estimator, os.path.join(save_dir, best_name))

    joblib.dump(gs, os.path.join(save_dir, filename))

    return joblib.load(os.path.join(save_dir, filename))


def main():
    df_orig = load_data(os.path.join(DATA_DIR, "240313_rc_steady.csv"))

    columns = list(df_orig)
    targets = ["loop-tof"]
    non_features = ["steady-state-condition"]
    features = [f for f in columns if f not in targets + non_features]

    target = targets[0]

    df = map_target_values(
        df_orig,
        target=target,
        target_name="log-loop-tof",
        encoding=lambda x: np.log(np.abs(x)) if x != 0 else 0,
    )

    X_train, X_test, y_train, y_test = split_train_test(
        df,
        target="log-loop-tof",
        features=features,
        test_size=0.10,
        scale_features=False,
        scale_target=False,
        scaler=StandardScaler(),
        random_state=SEED,
        check_shape=True,
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        booster="gbtree",
        random_state=SEED,
    )
    cv = StratifiedKFold(n_splits=10, shuffle=False)
    param_dists = {
        "n_estimators": range(100, 1500, 100),
        "min_child_weight": st.uniform(loc=1, scale=9),
        "max_depth": range(3, 12),
        "learning_rate": st.uniform(loc=0.001, scale=0.299),
        "gamma": st.uniform(loc=0, scale=5),
        "subsample": st.uniform(loc=0.5, scale=0.5),
        "colsample_bytree": st.uniform(loc=0.5, scale=0.5),
        "reg_alpha": st.uniform(loc=0.001, scale=100),
        "reg_lambda": st.uniform(loc=0.001, scale=100),
    }

    rs = search_random_params(
        X_train,
        y_train,
        model,
        param_dists,
        cv=cv,
        n_iters=5000,
        n_jobs=128,
        verbose=1,
        scoring="neg_mean_absolute_error",
        keep_trian_score=True,
        filename="xgb_reg_rc-random-cv-not-scaled-target.pkl",
        save_dir=SAVE_DIR,
        save_best=True,
        best_name="xgb_reg_rc-best-estm-not-scaled-target.pkl",
        remake=True,
    )

    return None


if __name__ == "__main__":
    main()
