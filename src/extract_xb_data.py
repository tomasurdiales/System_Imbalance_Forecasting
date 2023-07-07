# -*- coding: utf-8 -*-
"""
-> All code to extract net cross-border nominations data from raw Excel file.
"""
import pickle

import numpy as np
import pandas as pd

from src import utils

# Get root directory:
CWD = utils.get_root_dir()


def extract_from_raw_XB_files() -> pd.DataFrame:
    """
    -> Extracts data from raw excel file, cleaning and parsing datetimes.
    """
    # path_to_file = CWD / "data" / "XB_Nominations_and_flows.xlsx"
    path_to_file = CWD / "data" / "XB_Data_ODS.xlsx"
    print(f"Extracting XB Nominations data from:\n{path_to_file}")

    # Read only column names to build dtype dictionary:
    column_names = pd.read_excel(path_to_file, sheet_name="Data", skiprows=4, nrows=0)
    dtype_dict = {
        col: np.float32 if col != "DateAndTime" else str for col in column_names
    }

    # Read in data:
    df = pd.read_excel(path_to_file, sheet_name="Data", skiprows=4, dtype=dtype_dict)

    # Clean up, parse and localize datetime index:
    if path_to_file == CWD / "data" / "XB_Nominations_and_flows.xlsx":
        df = (
            df.assign(
                datetime_cet=pd.to_datetime(
                    df["DateAndTime"].str.replace("*", "", regex=False), utc=False
                ).dt.tz_localize("CET", ambiguous="infer")
            )
            .set_index("datetime_cet")
            .asfreq("15min", method=None)
            .drop("DateAndTime", axis="columns")
            .sort_index()
        )
    elif path_to_file == CWD / "data" / "XB_Data_ODS.xlsx":
        df = (
            df.assign(
                datetime_cet=pd.to_datetime(
                    df["DateAndTime"].str.replace("*", "", regex=False), utc=True
                )
            )
            .set_index("datetime_cet")
            .tz_convert("CET")
            .asfreq("15min", method=None)
            .drop("DateAndTime", axis="columns")
            .sort_index()
        )

    # Use a more sensible naming convention:
    df.columns = ["_".join(x.strip().lower().split()) for x in df.columns]

    return df


def main():
    """
    -> Extracts XB data from source and pickles to external file.
    """
    # Run data extraction:
    df = extract_from_raw_XB_files()

    # Write to picke file:
    with open(CWD / "data" / "xb_historical_data.pkl", "wb") as f:
        pickle.dump(df, f)
    # Write to parquet file:
    df.to_parquet(
        CWD / "data" / "xb_historical_data.parquet",
        engine="pyarrow",
        compression="snappy",
    )

    print(
        f"\nStored XB data in pickle+parquet format at:\n{CWD / 'data' / 'ems_historical_data.pkl'}"
    )


if __name__ == "__main__":
    main()
