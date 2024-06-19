import os
import re
import pandas as pd

CSVS_DIR = os.path.join("..", "..", "data", "csvs")


def load_data(filename, data_dir=CSVS_DIR):
    """
    Args:
        filename (string): The name of the file to load.
        data_dir (string, optional): The name of the directory to load from. Defaults to DATA_DIR.

    Returns:
        pandas.DataFrame
    """
    return pd.read_csv(os.path.join(data_dir, filename))


def save_data(df, filename, data_dir=CSVS_DIR):
    """
    Args:
        df (pandas.DataFrame): The dataframe to save.
        filename (string): The name of the file to save.
        data_dir (string, optional): The name of the directory to save to. Defaults to DATA_DIR.
    """
    return df.to_csv(os.path.join(data_dir, filename), index=False)


def rename_columns(df, column_names):
    """
    Args:
        df (pandas.DataFrame): The dataframe to rename columns in.
        column_names (dict): {old column names : new column names}.

    Returns:
        pandas.DataFrame
    """
    return df.rename(columns=column_names)


def remove_columns(df, column_names):
    """
    Args:
        df (pandas.DataFrame): The dataframe to remove columns from.
        column_names (list): A list of column names to remove.

    Returns:
        pandas.DataFrame
    """
    return df.drop(columns=column_names)


def split_steady_transient(df_orig):
    """

    Splits the dataframe into transient and steady state.

    Args:
        df (pandas.DataFrame): The dataframe to split.

    Returns:
        df_steady (pandas.DataFrame): The steady state data.
        df_transient (pandas.DataFrame): The transient state data.
    """

    df = df_orig.copy()
    df_steady = df[df["steady-state-condition"] == True]
    df_transient = df[df["steady-state-condition"] == False]

    return df_steady, df_transient


def main():

    filenames = ["final_simulation_outputs.csv", "final_simulation_outputs_rc.csv"]
    tags = ["op", "rc"]

    shared_cols_to_remove = [
        "simulation-id",
        "frequency",
    ]
    op_additional_cols_to_remove = [
        "bea",
    ]
    rc_additional_cols_to_remove = []

    oc_cols_to_remove = shared_cols_to_remove + op_additional_cols_to_remove
    rc_cols_to_remove = shared_cols_to_remove + rc_additional_cols_to_remove
    cols_to_remove = [oc_cols_to_remove, rc_cols_to_remove]

    zipped_specs = zip(filenames, cols_to_remove, tags)

    for filename, removals, tag in zipped_specs:
        df_orig = load_data(filename)
        df = df_orig.copy()

        col_names = list(df.columns)

        """
        Some funny syntax below.. re.sub is a regex function that replaces the first argument with the second argument in the third argument.
        The "-[\(\[].*?[\)\]]" part is a regex pattern that matches any string that starts with a hyphen, followed by an open parenthesis or bracket, followed by any characters, followed by a close parenthesis or bracket.
        Its just a generic way to remove the units from the column names.
        """
        col_aliases = {
            col: re.sub(
                "-[\(\[].*?[\)\]]",
                "",
                col.lower().replace(" ", "-").replace("_bea", "change-in-bea"),
            )
            for col in col_names
        }

        df = rename_columns(df, col_aliases)
        df = remove_columns(df, removals)

        # Catch any values of the TOF that didn't get set to 0 by Julia
        df["loop-tof"] = df["loop-tof"].apply(lambda x: 0 if -1e-4 < x < 1e-4 else x)

        df_steady, df_transient = split_steady_transient(df)

        save_data(df_steady, f"ml_data_{tag.lower()}_steady.csv")
        save_data(df_transient, f"ml_data_{tag.lower()}_transient.csv")

    return None


if __name__ == "__main__":
    main()
