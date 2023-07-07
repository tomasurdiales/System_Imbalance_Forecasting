# -*- coding: utf-8 -*-
"""
-> Python API to load dataset from OpenData Elia.
"""

from typing import Optional, Tuple

import pandas as pd
import requests  # type: ignore

parameters_default: dict = {
    "dataset": "ods046",  # imbalance prices per minute
    "sort": "datetime",
    "timezone": "utc",
    "rows": 300,
}


def get_dataset(parameters: dict | None = None) -> Tuple[pd.DataFrame, Optional[dict]]:
    return_dict = False
    if parameters is None:
        parameters = parameters_default
        return_dict = True

    # Load data from OpenData Elia:
    base_url = "https://opendata.elia.be/api/records/1.0/search/"
    response_json = requests.get(base_url, parameters, timeout=50).json()
    records = response_json["records"]
    data_list = []
    for item in records:
        fields = item["fields"]
        data_list.append(fields)

    print(
        f"Loaded {parameters['rows']} observations of the '{parameters['dataset']}' dataset."
    )
    if return_dict:
        return pd.DataFrame(data_list), parameters_default.copy()
    else:
        return pd.DataFrame(data_list)


def main():
    pass


if __name__ == "__main__":
    main()
