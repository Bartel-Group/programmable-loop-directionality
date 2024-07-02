import os
import joblib
import dice_ml
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

SEED = 17
CSVS_DIR = os.path.join("..", "..", "data", "csvs")
PKLS_DIR = os.path.join("..", "..", "data", "pkls")
FIGS_DIR = os.path.join("..", "..", "figures", "figure5")


def load_csv(filename, data_dir=CSVS_DIR):
    return pd.read_csv(os.path.join(data_dir, filename))


def load_model(filename, data_dir=PKLS_DIR):
    return joblib.load(os.path.join(data_dir, filename))


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


def init_dice(
    df, model, features, target="direction", backend="sklearn", model_type="classifier"
):
    d = dice_ml.Data(dataframe=df, continuous_features=features, outcome_name=target)
    m = dice_ml.Model(model=model, backend=backend, model_type=model_type)
    exp = dice_ml.Dice(d, m)

    return exp


def split_by_direction(df):
    """
    Args:
        df (pandas.DataFrame): The DataFrame to split

    Returns:
        pandas.DataFrame, pandas.DataFrame, pandas.DataFrame: DataFrames with positive, negative, and zero TOFs
    """
    positive_df = df[df["direction"] == 1]
    negative_df = df[df["direction"] == 2]
    zero_df = df[df["direction"] == 0]
    return positive_df, negative_df, zero_df


def generate_counterfactual_perturbations(query_instance, exp, num_cfs, desired_class):

    dice_exp = exp.generate_counterfactuals(
        query_instance, total_CFs=num_cfs, desired_class=desired_class
    )

    df_counterfactuals = dice_exp.cf_examples_list[0].final_cfs_df
    df_counterfactuals.drop(columns=["direction"], inplace=True)

    perturbations = (df_counterfactuals.sub(query_instance.squeeze())) / (
        query_instance.squeeze()
    )
    perturbations_abs = perturbations.abs()
    average_perturbations = perturbations_abs.mean(axis=0)
    np.array(average_perturbations).T

    return df_counterfactuals, average_perturbations, query_instance


def main():

    model_type = ["clf"]
    data_type = ["op"]

    for m in model_type:
        for d in data_type:
            model_file = f"{d}_steady/xgb_{m}_{d}.pkl"
            data_file = f"ml_data_{d}_steady.csv"

            model = load_model(model_file)
            df_orig = load_csv(data_file)

            columns = list(df_orig)
            target = "direction"
            non_features = ["steady-state-condition", "loop-tof"]
            features = [f for f in columns if f not in non_features]

            df_map = map_target(
                df_orig,
                target="loop-tof",
                new_target_name="direction",
                encoding=lambda x: (
                    1
                    if float("inf") > x >= 1e-4
                    else 2 if -float("inf") < x <= -1e-4 else 0
                ),
            )

            df = df_map.drop(columns=non_features, axis=1)

            train_dataset, test_dataset, _, _ = train_test_split(
                df, df[target], test_size=0.1, random_state=SEED
            )
            x_train, x_test = train_dataset.drop(
                columns=target, axis=1
            ), test_dataset.drop(columns=target, axis=1)

            exp = init_dice(df=train_dataset, model=model, features=features)
            pos_df, neg_df, zero_df = split_by_direction(test_dataset)

            start_dfs = [pos_df, neg_df, zero_df]
            start_classes = [1, 2, 0]
            desired_classes = [2, 1, 1]

            zipped = zip(start_dfs, start_classes, desired_classes)

            for df, start_class, desired_class in zipped:
                df = df.sample(frac=1).drop(columns=target, axis=1)
                length = len(df)

                query_inst_array = np.zeros((length, len(features)))
                counterfactual_array = np.zeros((10 * length, len(features)))
                perturbation_array = np.zeros((length, len(features)))

                for i in range(length):
                    (
                        counterfactual_array[(i * 10) : ((i + 1) * 10), :],
                        perturbation_array[i, :],
                        query_inst_array[i, :],
                    ) = generate_counterfactual_perturbations(
                        df[i : i + 1], exp, num_cfs=10, desired_class=desired_class
                    )

                perturbation_df = pd.DataFrame(perturbation_array, columns=features)
                counterfactual_df = pd.DataFrame(counterfactual_array, columns=features)
                query_inst_df = pd.DataFrame(query_inst_array, columns=features)

                data_file = os.path.join(
                    CSVS_DIR,
                    f"counterfactual-perturbations-{start_class}-to-{desired_class}.csv",
                )
                perturbation_df.to_csv(data_file1, index=False)

    return None


if __name__ == "__main__":
    main()
