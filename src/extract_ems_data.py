# -*- coding: utf-8 -*-
"""
-> All code to extract EMS data from raw files.
"""
import pickle

import numpy as np
import pandas as pd

from src import utils

# Get root directory:
CWD = utils.get_root_dir()


def extract_from_raw_EMS_files() -> pd.DataFrame:
    """
    -> Extracts data from raw EMS files, cleaning and parsing datetimes, and joining all into a single dataframe.
    """
    print(f"Extracting EMS minute-wise data from:\n{CWD / 'data' / 'EMS'}")

    # Obtain list of all files in data/EMS directory:
    all_files = [x for x in (CWD / "data" / "EMS").glob("*.csv") if x.is_file()]

    # Create a placeholder for all data:
    df_all = pd.DataFrame()

    # Iterate over every individual .csv file:
    for file_dir in all_files:
        # Read-in individual file, skipping the first two rows of trivial metadata:
        # *The decimal separator in the raw files is ',' so this needs to be specified while reading:
        df = pd.read_csv(file_dir, sep=";", skiprows=2, decimal=",")

        # Use a more sensible naming convention:
        df.columns = ["_".join(x.strip().lower().split()) for x in df.columns]

        # Re-cast all as float32:
        for column in [
            x
            for x in df.columns
            if x not in ["step_date_time", "elia_sum_prod_mw_coal_pulv_cm-value"]
        ]:
            df[column] = df[column].astype(np.float32)
        df["elia_sum_prod_mw_coal_pulv_cm-value"] = df[
            "elia_sum_prod_mw_coal_pulv_cm-value"
        ].astype(np.float32)

        # Re-index to the proper (timezone-aware) datetime vector:
        df = (
            df.assign(
                datetime_cet=pd.to_datetime(
                    df["step_date_time"], dayfirst=True, utc=False
                ).dt.tz_localize("CET", ambiguous="infer")
            )
            .set_index("datetime_cet")
            .asfreq("1min", method=None)
            .drop("step_date_time", axis="columns")
            .sort_index()
        )

        # Store individual file into the common dataframe:
        df_all = pd.concat([df_all, df])

    # Sort all concatenated data according to datetime:
    df_all = df_all.sort_index()

    # Remove duplicate entries at the beginning of each month:
    df_all = df_all[~df_all.index.duplicated(keep="first")]
    # Set frequency for entire dataset:
    df_all = df_all.asfreq("1T", method=None)

    return df_all


def main():
    """
    -> Extracts EMS data from source and pickles to external file.
    """
    # Run data extraction:
    df = extract_from_raw_EMS_files()

    # Write to picke file:
    with open(CWD / "data" / "ems_historical_data.pkl", "wb") as f:
        pickle.dump(df, f)
    # Write to parquet file:
    df.to_parquet(
        CWD / "data" / "ems_historical_data.parquet",
        engine="pyarrow",
        compression="snappy",
    )

    print(
        f"\nStored EMS minute-wise data in pickle+parquet format at:\n{CWD / 'data' / 'ems_historical_data.pkl'}"
    )


if __name__ == "__main__":
    main()
