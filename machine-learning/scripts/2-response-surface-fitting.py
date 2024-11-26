import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

SEED = 17
TRAINING_DATA_DIR = os.path.join("..", "training-data")


def load_csv(filename, data_dir=TRAINING_DATA_DIR):
    """
    Args:
        filename (string): The name of the file to load. (.csv)
        data_dir (string, optional): The name of the directory to load from. Defaults to TRAINING_DATA_DIR.

    Returns:
        pandas.DataFrame
    """
    return pd.read_csv(os.path.join(data_dir, filename))


def map_target_values(df, target, new_target_name, encoding):
    """
    Args:
        df (pandas.DataFrame): The dataframe to map the target values.
        target (string): The name of the target column.
        new_target_name (string): The name of the new target column.
        encoding (lambda function): A lambda function to map the target values.
            e.g. lambda x: np.log(np.abs(x))

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
    tags = ["op", "rc"]

    for tag in tags:

        df_orig = load_csv(filename="ml_data_" + tag + "_steady.csv")
        df_copy = df_orig.copy()

        columns = list(df_copy)
        targets = ["loop-tof"]
        non_features = ["steady-state-condition"]
        features = [f for f in columns if f not in targets + non_features]

        condition = (df_copy["loop-tof"] <= -1e-4) | (df_copy["loop-tof"] >= 1e-4)
        df_filtered = df_copy.loc[condition]

        df = map_target_values(
            df_filtered,
            target="loop-tof",
            new_target_name="log-loop-tof",
            encoding=lambda x: np.log(np.abs(x)),
        )

        X_train, X_test, Y_train, Y_test = split_train_test(
            df,
            target="log-loop-tof",
            features=features,
            test_size=0.10,
            random_state=SEED,
            check_shape=False,
        )

        Y_train, Y_test = Y_train.ravel(), Y_test.ravel()

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        poly_features_train = PolynomialFeatures(degree=2, include_bias=False)
        X_poly_train = poly_features_train.fit_transform(X_train_scaled)

        model = LinearRegression(fit_intercept=False)
        model.fit(X_poly_train, Y_train)

        poly_features_test = PolynomialFeatures(degree=2, include_bias=False)
        X_poly_test = poly_features_test.fit_transform(X_test_scaled)

        y_pred = model.predict(X_poly_test)
        y = np.exp(y_pred)

        y_test = np.abs(np.array(Y_test.flatten()))

        rmse = np.sqrt(mean_squared_error(y_test, y))
        mae = mean_absolute_error(y_test, y)
        median_abs_err = np.median(np.abs(y_test - y))
        seventyfifth_percentile = np.percentile(np.abs(y_test - y), 75)

        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"Median AE: {median_abs_err}")
        print(f"75th Percentile AE: {seventyfifth_percentile}")
        print("------------------")

    return None


if __name__ == "__main__":
    main()
