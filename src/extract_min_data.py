# -*- coding: utf-8 -*-
"""
-> All code to extract minute-wise data from raw files.
"""

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import typer

from src import utils

# Get root directory:
CWD = utils.get_root_dir()


def extract_from_raw_opendata(
    path_to_file: str | os.PathLike | None = None,
    downcast: bool | None = True,
    fill_missing: bool | None = True,
    keep_all_columns: bool | None = False,
    verbose: bool | None = False,
) -> pd.DataFrame:
    """
    -> Extracts data from raw Elia OpenData file, cleaning and parsing datetimes.
    """

    # Set path to raw file:
    if path_to_file is None:
        path_to_file = CWD / "data" / "ods046.csv"

    # Read only column names to build dtype dictionary:
    column_names = pd.read_csv(path_to_file, sep=";", nrows=0)
    if downcast:
        dtype_dict = {
            col: np.float32
            if col
            not in [
                "Datetime",
                "Resolution code",
                "Quarter hour",
                "Quality status",
                "Calculation time",
            ]
            else str
            for col in column_names
        }
    else:
        dtype_dict = {
            col: np.float64
            if col
            not in [
                "Datetime",
                "Resolution code",
                "Quarter hour",
                "Quality status",
                "Calculation time",
            ]
            else str
            for col in column_names
        }
    # And keep quality status as categorical variable:
    dtype_dict["Quality status"] = "category"

    # Read whole file. *Specifying dtypes manually to avoid 'converters' warning in pd.read_csv
    print(f"Extracting minute-wise data from:\n{path_to_file}")
    if downcast:
        print("*Downcasting floats to 32bit\n")
    df = pd.read_csv(path_to_file, sep=";", dtype=dtype_dict)

    # Parse datetimes and set datetime index:
    # *Needs to be read in as UTC and then can be converted/localized back to CET. Otherwise pandas does not know how to parse it as a datetime index.
    df = (
        df.assign(datetime_cet=pd.to_datetime(df["Datetime"], utc=True))
        .set_index("datetime_cet")
        .tz_convert("CET")
        .sort_index()
        .drop("Datetime", axis="columns")
    )

    # Set frequency and fill in missing datetimes:
    if fill_missing:
        df = df.asfreq("1min", method=None)
        missing_entries = df.index[df.isnull().all(axis="columns")]
        if not missing_entries.empty:
            print(
                f"WARNING: {missing_entries.size} missing entries in datetime index were filled with NaN!"
            )
            if verbose:
                print(missing_entries)
    else:
        print(
            "WARNING: Missing datetimes are not accounted for and the frequency has not been set!"
        )

    # Drop useless columns:
    if not keep_all_columns:
        to_drop = [
            "Resolution code",
            "Quarter hour",
            #    "Quality status",
            "Calculation time",
            "Strategic reserve price",
            "System imbalance vs Incremental bids coordinable",
        ]
        df = df.drop(to_drop, axis="columns")
    else:
        print("\n*Keeping all original columns.")

    # Apply a more sensible naming convention:
    # *Removes any unnecessary spaces, converts all to lowercase, and joins by _ instead of spaces
    df.columns = ["_".join(x.strip().lower().split()) for x in df.columns]

    # Keep only the years used in modelling: 2021-2022
    df = df.loc["2021":"2022"]  # type: ignore

    print("\nSuccessfully extracted data.")
    return df


def main(
    path_to_file: Optional[str] = None,
    downcast: Optional[bool] = True,
    fill_missing: Optional[bool] = True,
    keep_all_columns: Optional[bool] = False,
    verbose: Optional[bool] = False,
):
    df = extract_from_raw_opendata(
        path_to_file=path_to_file,
        downcast=downcast,
        fill_missing=fill_missing,
        keep_all_columns=keep_all_columns,
        verbose=verbose,
    )

    # Write to picke file:
    with open(CWD / "data" / "min_historical_data.pkl", "wb") as f:
        pickle.dump(df, f)
    # Write to parquet file:
    df.to_parquet(
        CWD / "data" / "min_historical_data.parquet",
        engine="pyarrow",
        compression="snappy",
    )

    if verbose:
        print(
            f"\nStored minute-wise data in pickle+parquet format at:\n{CWD / 'data' / 'min_historical_data.pkl'}"
        )


if __name__ == "__main__":
    typer.run(main)
