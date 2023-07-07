# -*- coding: utf-8 -*-

import pandas as pd

from src import utils


def join_qh_min_data(
    minute: int | str,
    qh_data: pd.DataFrame,
    qh_parameters: dict,
    minute_data: pd.DataFrame | None = None,
    minute_parameters: dict | None = None,
    ems_data: pd.DataFrame | None = None,
    ems_parameters: dict | None = None,
) -> pd.DataFrame:
    """
    -> Merge quarter-hourly and minute-wise extracted datasets
    """

    if type(minute) is str and minute != "all":
        print("ERROR: provided a string for a numerical value of MINUTE!")

    # Generate dataframes with desired lagged features:
    qh_data = utils.generate_lagged_features(data=qh_data, parameters=qh_parameters)
    # Turn quarter-hourly dataset into "fake" minute frequency for merging with minute-wise dataset:
    qh_data = qh_data.asfreq("1min", method="ffill")

    if (minute_data is not None) and (minute_parameters is not None):
        minute_data = utils.generate_lagged_features(
            data=minute_data, parameters=minute_parameters
        )
        # Merge them:
        df = pd.concat([qh_data, minute_data], axis="columns", join="inner")
    else:
        df = qh_data

    # Add in EMS data if desired:
    if (ems_data is not None) and (ems_parameters is not None):
        ems_data = utils.generate_lagged_features(
            data=ems_data, parameters=ems_parameters
        )
        df = df.merge(ems_data, how="inner", right_index=True, left_index=True)

    # Filter for just the current qh minute:
    if minute != "all":
        df = df.loc[df.index.minute % 15 == minute]

    return df


def main():
    pass


if __name__ == "__main__":
    main()
