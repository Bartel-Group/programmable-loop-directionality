import os
import joblib
import pandas as pd
import scipy.stats as st

from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    train_test_split,
)

from xgboost import XGBClassifier

SEED = 17  # <- The best number
DATA_DIR = "/Users/noord014/MyDrive/phd/research/dmc-lab/bartel-group/programmable-catalysis/dev/data/csvs"


def load_data(filename, data_dir=DATA_DIR):
    """
    Args:
        filename (string): The name of the file to load. (.csv)
        data_dir (string, optional): The name of the directory to load from. Defaults to DATA_DIR.

    Returns:
        pandas.DataFrame
    """
    return pd.read_csv(os.path.join(data_dir, filename))


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
            """train_test_split might throw an error if the shapes are different but it's good to check"""
            raise ValueError("X_train and y_train have different number of samples")
        else:
            return X_train, X_test, y_train, y_test

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
    scoring="f1_weighted",
    keep_trian_score=False,
    filename="xgb.pkl",
    save_dir=DATA_DIR,
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
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to 1. (n_jobs=n_cores=n_threads=n_tasks or whatever other names it has...)
        verbose (int, optional): Controls the verbosity. Defaults to 2.
        scoring (string, callable, list/tuple, dict or None, optional): A single string to evaluate the predictions on the test set. Defaults to "f1_weighted".
        keep_trian_score (bool, optional): If False, the cv_results_ attribute will not include training scores. Defaults to False for memory efficiency.
        filename (string, optional): The name of the file to save the search. Defaults to "xgb.pkl".
        save_dir (string, optional): The name of the directory to save the search. Defaults to DATA_DIR.
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

    joblib.dump(gs, os.path.join(save_dir, filename))

    return joblib.load(os.path.join(save_dir, filename))


def main():
    df_orig = load_data(os.path.join(DATA_DIR, "240219_op_steady.csv"))

    columns = list(df_orig)
    targets = ["loop-tof"]
    non_features = ["steady-state-condition"]
    features = [f for f in columns if f not in targets + non_features]

    target = targets[0]

    df = map_target_classes(
        df_orig, target=target, encoding=lambda x: 1 if x > 0 else 2 if x < 0 else 0
    )

    X_train, X_test, y_train, y_test = split_train_test(
        df,
        target="class",
        features=features,
        test_size=0.10,
        random_state=SEED,
        check_shape=True,
    )

    model = XGBClassifier(
        objective="multi:softmax",
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
        n_iters=100,
        n_jobs=4,
        verbose=1,
        scoring="f1_weighted",
        keep_trian_score=True,
        filename="xgb_clf_op-random-cv.pkl",
        save_dir=os.path.join(DATA_DIR, "pkls"),
        remake=True,
    )

    return None


if __name__ == "__main__":
    main()
