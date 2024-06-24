import os
import joblib
import pandas as pd
import scipy.stats as st

from xgboost import XGBClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    train_test_split,
)

SEED = 17
CSVS_DIR = os.path.join("..", "..", "data", "csvs")
PKLS_DIR = os.path.join("..", "..", "data", "pkls")


def load_csv(filename, data_dir=CSVS_DIR):
    """
    Args:
        filename (string): The name of the file to load. (.csv)
        data_dir (string, optional): The name of the directory to load from. Defaults to CSVS_DIR.

    Returns:
        pandas.DataFrame
    """
    return pd.read_csv(os.path.join(data_dir, filename))


def map_target_classes(df, target, new_target_name, encoding):
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


def search_random_params(
    X_train,
    Y_train,
    model,
    param_dists,
    cv=5,
    n_iters=100,
    n_jobs=1,
    verbose=2,
    scoring="f1_weighted",
    random_state=SEED,
    keep_train_score=False,
    filename="xgb_clf.pkl",
    save_dir=PKLS_DIR,
    save_best=True,
    best_name="xgb_clf_best-estm.pkl",
    remake=False,
):
    """
    Args:
        X_train (array-like): The training input samples.
        Y_train (array-like): The target values.
        model (sklearn.base.BaseEstimator): The model to fit.
        param_dists (dict): Dictionary with parameters names (string) as keys and distributions or lists of parameters to try.
        cv (int, cross-validation generator or an iterable, optional): Determines the cross-validation splitting strategy. Defaults to 5.
        n_iters (int, optional): Number of parameter settings that are sampled. Defaults to 100.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to 4. (n_jobs=n_cores=n_tasks etc..)
        verbose (int, optional): Controls the verbosity. Defaults to 2.
        random_state (int, RandomState instance or None, optional): Controls the randomness of the estimator. Defaults to SEED.
        scoring (string, callable, list/tuple, dict or None, optional): A single string (see The scoring parameter: defining model evaluation rules) or a callable (see Defining your scoring strategy from metric functions) to evaluate the predictions on the test set. Defaults to "f1_weighted".
        keep_train_score (bool, optional): If False, the cv_results_ attribute will not include training scores. Defaults to False for memory efficiency.
        filename (string, optional): The name of the file to save the search. Defaults to "xgb_clf.pkl".
        save_dir (string, optional): The name of the directory to save the search. Defaults to PKLS_DIR.
        save_best (bool, optional): If True, will save the best estimator separatly. Defaults to True.
        best_name (string, optional): The name of the file to save the best estimator. Defaults to "xgb_clf_best-estm.pkl".
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
        random_state=random_state,
        return_train_score=keep_train_score,
    )
    gs.fit(X_train, Y_train)

    if save_best:
        best_estimator = gs.best_estimator_
        joblib.dump(best_estimator, os.path.join(save_dir, best_name))

    joblib.dump(gs, os.path.join(save_dir, filename))
    return joblib.load(os.path.join(save_dir, filename))


def main():
    tags = ["op", "rc"]

    for tag in tags:

        pkls_subdir = os.path.join(PKLS_DIR, tag + "_steady")

        df_orig = load_csv(filename="ml_data_" + tag + "_steady.csv")

        columns = list(df_orig)
        targets = ["loop-tof"]
        non_features = ["steady-state-condition"]
        features = [f for f in columns if f not in targets + non_features]

        df = map_target_classes(
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
            Y_train,
            model,
            param_dists,
            cv=cv,
            n_iters=5000,
            n_jobs=128,
            verbose=1,
            scoring="f1_weighted",
            keep_train_score=True,
            filename="xgb_clf_" + tag + ".pkl",
            save_dir=pkls_subdir,
            save_best=True,
            best_name="xgb_clf_" + tag + "-best-estm.pkl",
            remake=True,
        )

    return None


if __name__ == "__main__":
    main()
