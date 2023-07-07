# -*- coding: utf-8 -*-
"""
-> All code to extract quarter-hourly data from raw files.
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


def extract_from_raw_datawarehouse(
    path_to_file: str | os.PathLike | None = None,
    downcast: bool | None = True,
    keep_all: bool | None = True,
    verbose: bool | None = False,
) -> pd.DataFrame:
    """
    -> Extracts data from raw DataWarehouse Excel, cleaning and parsing datetimes.
    """

    # Set path to raw file:
    if path_to_file is None:
        path_to_file = CWD / "data" / "SourceData_v02.xlsb"

    # Read only column names to build dtype dictionary:
    column_names = pd.read_excel(path_to_file, sheet_name="All", skiprows=3, nrows=0)
    if downcast:
        dtype_dict = {
            col: np.float32 if col != "DateAndTime" else str for col in column_names
        }
    else:
        dtype_dict = {
            col: np.float64 if col != "DateAndTime" else str for col in column_names
        }

    # Read whole file. *Specifying dtypes manually to avoid 'converters' warning in pd.read_excel
    print(f"Extracting quarter-hourly data from:\n{path_to_file}")
    if downcast:
        print("*Downcasting floats to 32bit")
    df = pd.read_excel(path_to_file, sheet_name="All", skiprows=3, dtype=dtype_dict)

    # To properly parse data with a complete datetime index, we'll have to manually reindex:
    # *Start by taking a copy while removing DST entries with incompatible string form (*):
    validation_df = df[~df["DateAndTime"].str.contains("*", regex=False)].copy()

    # Scan for Daylight Saving Time entries (*):
    if any(df["DateAndTime"].str.contains("*", regex=False)):
        print(
            f"\nWARNING: {sum(df['DateAndTime'].str.contains('*', regex=False))} DST (*) entries in datetime index!"
        )
        if verbose:
            print(
                df["DateAndTime"][
                    df["DateAndTime"].str.contains("*", regex=False)
                ].values
            )

    # Clean and parse datetimes on this validation dataset:
    validation_df = (
        validation_df.assign(
            datetime=pd.to_datetime(
                validation_df["DateAndTime"].str.replace("*", "", regex=False)
            )
        )
        .set_index("datetime")
        .drop("DateAndTime", axis="columns")
        .sort_index()
    )

    # Scan for missing entries:
    if not pd.date_range(
        start=validation_df.index.min(), end=validation_df.index.max(), freq="15min"
    ).equals(validation_df.index):
        missing = pd.date_range(
            start=validation_df.index.min(), end=validation_df.index.max(), freq="15min"
        ).difference(validation_df.index)

        print(f"WARNING: {missing.size} missing entries in datetime index!")
        if verbose:
            print(missing, "\n")

    # Now try to reindex original dataset with a complete CET datetime index:
    #  *Do this by generating a full index starting on the first datetime for len(df) periods.
    df = (
        df.assign(
            datetime_cet=pd.date_range(
                start=validation_df.index.min(), tz="CET", freq="15min", periods=len(df)
            )
        )
        .set_index("datetime_cet")
        .asfreq("15min", method=None)
        .drop("DateAndTime", axis="columns")
        .sort_index()
    )

    # Use validation_df to check that data is consistent after reindexing:
    # *Do this by generating a set of random days where we check for consistency. Needs to be done this way becuase tz-naive datetime index will not work with tz-naive, and viceversa.
    for date in pd.Series(
        pd.date_range(
            validation_df.index.min(), end=validation_df.index.max(), freq="15min"
        )
    ).sample(1000):
        if date.month not in [3, 10]:  # (avoid March and October)
            date_str = str(
                date.date()
            )  # holds a string with a random date of the year, so that it is generic enough to work with both .loc[]
            if (
                df.loc[date_str]
                .reset_index(drop=True)
                .equals(validation_df.loc[date_str].reset_index(drop=True))
            ):
                continue
            else:
                print(f"\nERROR: Found a mismatch of values on day {date_str}!")
                break

    # Assign corrected WIND_RT column (modified by Jose) to its original name:
    if "WIND_CORRECTED RT_MW" in df.columns:
        df = df.drop(columns="WIND_RT_MW").rename(
            columns={"WIND_CORRECTED RT_MW": "WIND_RT_MW"}
        )
        print("\n*Changed 'WIND_CORRECTED RT_MW' to 'WIND_RT_MW'")

    # Apply a more sensible naming convention:
    # *Removes any unnecessary spaces, converts all to lowercase, and joins by _ instead of spaces
    df.columns = ["_".join(x.strip().lower().split()) for x in df.columns]

    # Keep only the years used in modelling: 2021-2022
    if not keep_all:
        df = df.loc["2021":"2022"]  # type: ignore

    # Extract system_imbalance_cum15 from min_historical_data.pkl:
    minute = utils.load_min_historical_data()
    system_imbalance_cum15 = (
        minute["system_imbalance"].loc[minute.index.minute % 15 == 14].copy()
    )
    # We need to round/reset the minutes and re-index the whole Series:
    system_imbalance_cum15.index = pd.date_range(
        start=system_imbalance_cum15.index.min().round(freq="H"),
        tz="CET",
        freq="15min",
        periods=len(system_imbalance_cum15),
    )
    # And insert into main dataframe:
    df.insert(1, "system_imbalance_cum15", system_imbalance_cum15)

    print("\nSuccessfully extracted data.")
    return df


def main(
    path_to_file: Optional[str] = None,
    downcast: Optional[bool] = True,
    keep_all: Optional[bool] = True,
    verbose: Optional[bool] = False,
):
    """
    -> Extracts qh data from source and pickles to external file.
    """
    # Extract:
    df = extract_from_raw_datawarehouse(
        path_to_file=path_to_file,
        downcast=downcast,
        keep_all=keep_all,
        verbose=verbose,
    )

    # Write to picke file:
    with open(CWD / "data" / "qh_historical_data.pkl", "wb") as f:
        pickle.dump(df, f)
    # Write to parquet file:
    df.to_parquet(
        CWD / "data" / "qh_historical_data.parquet",
        engine="pyarrow",
        compression="snappy",
    )

    if verbose:
        print(
            f"\nStored quarter-hourly data in pickle+parquet format at:\n{CWD / 'data' / 'qh_historical_data.pkl'}"
        )


if __name__ == "__main__":
    typer.run(main)
