# -*- coding: utf-8 -*-
"""
-> Common tools used throughout project.
"""

import pathlib
import sys
from platform import python_version

import numpy as np
import pandas as pd
from scipy.stats import t

# ----------------------------------------------------------------------------------------------------------------------


# Quickly load data:
def load_qh_historical_data() -> pd.DataFrame:
    """
    -> Loads quarter-hourly historical records from pickle/parquet file.
    """
    # with open(get_root_dir() / "data" / "qh_historical_data.pkl", "rb") as f:
    #     df = pickle.load(f)
    df = pd.read_parquet(
        get_root_dir() / "data" / "qh_historical_data.parquet", engine="pyarrow"
    )
    df = df.asfreq(pd.infer_freq(df.index), method=None)
    return df


def load_min_historical_data() -> pd.DataFrame:
    """
    -> Loads minute-wise historical records from pickle/parquet file.
    """
    # with open(get_root_dir() / "data" / "min_historical_data.pkl", "rb") as f:
    #     df = pickle.load(f)
    df = pd.read_parquet(
        get_root_dir() / "data" / "min_historical_data.parquet", engine="pyarrow"
    )
    df = df.asfreq(pd.infer_freq(df.index), method=None)
    return df


def load_ems_historical_data() -> pd.DataFrame:
    """
    -> Loads EMS minute-wise data from pickle/parquet file.
    """
    # with open(get_root_dir() / "data" / "ems_historical_data.pkl", "rb") as f:
    # df = pickle.load(f)
    df = pd.read_parquet(
        get_root_dir() / "data" / "ems_historical_data.parquet", engine="pyarrow"
    )
    df = df.asfreq(pd.infer_freq(df.index), method=None)
    return df


def load_xb_historical_data() -> pd.DataFrame:
    """
    -> Loads cross-border nominations data from pickle/parquet file.
    """
    # with open(get_root_dir() / "data" / "xb_historical_data.pkl", "rb") as f:
    # df = pickle.load(f)
    df = pd.read_parquet(
        get_root_dir() / "data" / "xb_historical_data.parquet", engine="pyarrow"
    )
    df = df.asfreq(pd.infer_freq(df.index), method=None)
    return df


def load_temp_historical_data() -> pd.DataFrame:
    """
    -> Loads temperature nominations data from pickle/parquet file.
    """
    # with open(get_root_dir() / "data" / "xb_historical_data.pkl", "rb") as f:
    # df = pickle.load(f)
    df = pd.read_parquet(
        get_root_dir() / "data" / "temp_historical_data.parquet", engine="pyarrow"
    )
    df = df.asfreq(pd.infer_freq(df.index), method=None)
    return df


# ----------------------------------------------------------------------------------------------------------------------


def DM_test(
    y_test: pd.Series,
    y_pred_1: pd.Series,
    y_pred_2: pd.Series,
    h: int = 1,
    harvey_adj: bool = True,
):
    """
    -> Performs the Diebold-Mariano test to check for statistical significance in difference between two forecasts.

    Arguments:
        y_test -> real forecasted time-series
        y_pred_1 -> first forecast to compare
        y_pred_2 -> second forecast to compare
        h -> forecast horizon
        harvey_adj -> Harvey adjustment (leave as True)
    """

    e1_lst = []
    e2_lst = []
    d_lst = []

    y_test = y_test.tolist()
    y_pred_1 = y_pred_1.tolist()
    y_pred_2 = y_pred_2.tolist()

    # Length of forecasts
    T = float(len(y_test))

    # Construct loss differential according to error criterion (MSE)
    for real, p1, p2 in zip(y_test, y_pred_1, y_pred_2):
        e1_lst.append((real - p1) ** 2)
        e2_lst.append((real - p2) ** 2)
    for e1, e2 in zip(e1_lst, e2_lst):
        d_lst.append(e1 - e2)

    # Mean of loss differential
    mean_d = pd.Series(d_lst).mean()

    # Calculate autocovariance
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N - k):
            autoCov += ((Xi[i + k]) - Xs) * (Xi[i] - Xs)
        return (1 / (T)) * autoCov

    # Calculate the denominator of DM stat
    gamma = []
    for lag in range(0, h):
        gamma.append(autocovariance(d_lst, len(d_lst), lag, mean_d))  # 0, 1, 2
    V_d = (gamma[0] + 2 * sum(gamma[1:])) / T

    # Calculate DM stat
    DM_stat = V_d ** (-0.5) * mean_d

    # Calculate and apply Harvey adjustement
    # It applies a correction for small sample
    if harvey_adj is True:
        harvey_adj = ((T + 1 - 2 * h + h * (h - 1) / T) / T) ** (0.5)
        DM_stat = harvey_adj * DM_stat

    # Calculate p-value
    p_value = 2 * t.cdf(-abs(DM_stat), df=T - 1)

    print(f"DM Statistic: {DM_stat :.4f} | p-value: {p_value :.4f}")


def generate_lagged_features(data: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    -> Takes in original dataframe, selects usable input features according to data availability, and includes the chosen lags accordingly.

    Arguments:
        data -> input pd.DataFrame
        parameters -> dictionary containing lags and real-time availability for each desired input feature
        minute -> minute at which predictions are made

    Returns:
        pd.DataFrame with the desired
    """
    # Only use columns that are in parameters dictionary:
    data = data[parameters.keys()].copy()
    output_df = []

    # Start by figuring out whether data is for a single minute in the qh or for all minutes:
    guess_index = np.random.randint(len(data) - 1)
    flag_minute_data = (
        abs(data.index[-guess_index].minute - data.index[-guess_index - 1].minute) < 15
    )
    granularity = "minute" if flag_minute_data else "qh"

    if data.index.freq is None:
        print("WARNING: Frequency for this dataframe is undefined!")
        print(f"Frequency identified as: {granularity}")

    for column in data:
        for lag in parameters[column]["lags"]:
            # Set a descriptive name for this column
            if lag < 0:
                column_name = f"{column}_from_{granularity}_minus_{abs(lag)}"
            elif lag == 0:
                column_name = f"{column}_current_{granularity}"
            elif lag > 0:
                column_name = f"{column}_from_{granularity}_plus_{abs(lag)}"

            output_df.append(data[column].shift(-lag).rename(column_name))

    return pd.concat(output_df, axis="columns")


def create_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    -> Generate datetime features from index
    """
    df = df.copy().assign(
        minute=df.index.minute.values.astype(np.int16),
        hour=df.index.hour.values.astype(np.int16),
        dayofmonth=df.index.day.values.astype(np.int16),
        month=df.index.month.values.astype(np.int16),
        year=df.index.year.values.astype(np.int16),
        # dayofweek_name = df.index.day_name(),
        dayofweek=df.index.dayofweek.values.astype(np.int16),
        dayofyear=df.index.dayofyear.values.astype(np.int16),
        weekofyear=df.index.isocalendar().week.values.astype(np.int16),
    )
    return df


def check_datetime_index(df: pd.DataFrame, freq: str | None = None):
    """
    -> Checks whether a *quarter-hourly* datetime index has missing entries
    """
    if freq is None:
        freq = "15min"
    missing = pd.date_range(
        start=df.index.min(), end=df.index.max(), freq=freq, tz=df.index.tz
    ).difference(df.index)

    if missing.empty:
        print("No missing entries.")
    else:
        print(missing)


# ----------------------------------------------------------------------------------------------------------------------


# System level utils:
def get_root_dir() -> pathlib.Path:
    """
    -> Returns project root directory, regardless of where it's called from.
    """
    return pathlib.Path(__file__).resolve().parents[1]


def show_python_env():
    print(f"Python {python_version()}")
    print(f"On path:\n{sys.path}")


# ----------------------------------------------------------------------------------------------------------------------


def main():
    print(f"Root directory: {get_root_dir()}")
    show_python_env()


if __name__ == "__main__":
    main()
