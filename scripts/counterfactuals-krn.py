import dice_ml
from dice_ml import Dice
from dice_ml.utils import helpers  # helper functions

import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as st
import xgboost
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

SEED = 17
DATA_DIR = "/Users/noord014/MyDrive/phd/research/dmc-lab/bartel-group/programmable-catalysis/dev/data"
FIG_DIR = "/Users/noord014/MyDrive/phd/research/dmc-lab/bartel-group/programmable-catalysis/dev/figures"


def load_model(filename, data_dir=DATA_DIR):
    return joblib.load(os.path.join(data_dir, filename))


def load_data(filename, data_dir=DATA_DIR):
    data_file = os.path.join(data_dir, filename)
    return pd.read_csv(data_file)


def map_target_classes(df, target, encoding):
    df["direction"] = df[target].map(encoding)
    return df


def init_dice(
    df, model, features, target="direction", backend="sklearn", model_type="classifier"
):
    # identify targets and features
    d = dice_ml.Data(dataframe=df, continuous_features=features, outcome_name=target)

    # Pre-trained ML model
    m = dice_ml.Model(model=model, backend=backend, model_type=model_type)

    # DiCE explanation instance
    exp = dice_ml.Dice(d, m)

    return exp


def parse_outputs(df, target="direction"):
    # Group by direction
    grouped = df.groupby(target)

    # Split the df into separate dfs
    dfs = {class_label: group for class_label, group in grouped}

    # Access each DataFrame by class label
    df_zero = dfs[0]
    df_pos = dfs[1]
    df_neg = dfs[2]

    return df_zero, df_pos, df_neg


def generate_counterfactual_perturbations(query_instance, exp, num_cfs, desired_class):
    # Generate counterfactual examples

    query_instance = data.drop(columns="direction")
    # print(query_instance)

    dice_exp = exp.generate_counterfactuals(
        query_instance, total_CFs=num_cfs, desired_class=desired_class
    )

    df_counterfactuals = dice_exp.cf_examples_list[0].final_cfs_df
    df_counterfactuals.drop(columns=["direction"], inplace=True)

    # determine the feature perturbations
    perturbations = (df_counterfactuals.sub(query_instance.squeeze())) / (
        query_instance.squeeze()
    )
    perturbations_abs = perturbations.abs()
    average_perturbations = perturbations_abs.mean(axis=0)
    np.array(average_perturbations).T

    return df_counterfactuals, average_perturbations, query_instance


def main():
    model = load_model("pkls/rc_steady/xgb_clf_rc-best-estm.pkl")
    df_orig = load_data("csvs/240313_rc_steady.csv")

    df_map = map_target_classes(
        df_orig,
        target="loop-tof",
        encoding=lambda x: 1 if x > 0 else 2 if x < 0 else 0,
    )

    target = "direction"
    non_features = ["steady-state-condition", "loop-tof"]
    columns = list(df_map)
    features = [f for f in columns if f is not target if f not in non_features]
    columns_to_drop = [col for col in non_features if col in columns]

    df = df_map.drop(columns=columns_to_drop, axis=1)

    train_dataset, test_dataset, y_train, y_test = train_test_split(
        df, df[target], test_size=0.1, random_state=SEED
    )
    x_train, x_test = train_dataset.drop(columns=target, axis=1), test_dataset.drop(
        columns=target, axis=1
    )

    df, exp, features = init_dice(df=train_dataset, model=model, features=features)
    df_zero, df_pos, df_neg = parse_outputs(x_test)

    # shuffle df
    df_neg = df_neg.sample(frac=1)

    length = 250

    query_inst_array = np.zeros((length, len(features)))
    counterfactual_array = np.zeros((10 * length, len(features)))
    perturbation_array = np.zeros((length, len(features)))

    for i in range(length):
        (
            counterfactual_array[(i * 10) : ((i + 1) * 10), :],
            perturbation_array[i, :],
            query_inst_array[i, :],
        ) = generate_counterfactual_perturbations(
            df_neg[i : i + 1], exp, num_cfs=10, desired_class=1
        )

    # Convert array to DataFrame
    perturbation_df = pd.DataFrame(perturbation_array, columns=features)
    counterfactual_df = pd.DataFrame(counterfactual_array, columns=features)
    query_inst_df = pd.DataFrame(query_inst_array, columns=features)

    # Write DataFrame to CSV
    data_dir = "/home/dauenha0/murp1677/Cyclic_Dynamics/Code/ML_repo/programmable-catalysis/dev/data/"

    data_file1 = os.path.join(data_dir, "counterfactual-perturbations-neg-to-pos.csv")
    perturbation_df.to_csv(data_file1, index=False)

    data_file2 = os.path.join(data_dir, "counterfactuals-neg-to-pos.csv")
    counterfactual_df.to_csv(data_file2, index=False)

    data_file3 = os.path.join(data_dir, "query-instances-neg-to-pos.csv")
    query_inst_df.to_csv(data_file3, index=False)

    return None


if __name__ == "__main__":
    res = main()
