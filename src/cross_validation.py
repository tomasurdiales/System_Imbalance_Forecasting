# -*- coding: utf-8 -*-
"""
-> Script to train and cross-validate a forecasting model (scikit-learn compatible) on a provided TimeSeriesSplit.
"""
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def backtesting_CV(
    model,
    data: pd.DataFrame,
    time_splits: TimeSeriesSplit,
    features: list | str,
    target: list | str,
    use_scaler: bool | None = True,
    progress_bar: bool | None = True,
    print_error_metrics: bool | None = True,
    verbose: bool | None = False,
):
    """
    -> Trains a given model on the chosen time-split configuration to perform cross-validation through backtesting.
    *Uses StandardScaler for all its models.

    Arguments:
        model -> scikit-learn compatible model
        data -> input data
        time_splits -> TimeSeriesSplit (sklearn) configuration
        features -> data's input features
        target -> data's forecasting objective
        use_scaler -> whether to add StandardScaler to model pipeline
        progress_bar -> whether to display a progress bar
        print_error_metrics -> whether to print out all error measurements
        verbose -> whether to show MAE/RMSW for some folds while training

    Returns:
        [backtest_df, [error_metrics]]
    """
    start = perf_counter()
    model_name = type(model).__name__
    print(f"MODEL: {model_name}")

    # Start by figuring out whether data is for a single minute in the qh or for all minutes:
    guess_index = np.random.randint(len(data) - 1)
    all_minutes = (
        abs(data.index[-guess_index].minute - data.index[-guess_index - 1].minute) < 15
    )

    if all_minutes:
        print(
            f"Time configuration: {time_splits.n_splits} splits, {time_splits.test_size//15//4//24} testing days, \
{time_splits.max_train_size//15//4//24//7} training weeks. \
Total predicted time: {time_splits.n_splits*time_splits.test_size//15//4//24} days.\n"
        )
    else:
        print(
            f"Time configuration: {time_splits.n_splits} splits, {time_splits.test_size//4//24} testing days, \
{time_splits.max_train_size//4//24//7} training weeks. \
Total predicted time: {time_splits.n_splits*time_splits.test_size//4//24} days.\n"
        )

    # Set up output variables and flags:
    backtest, predictions, split_numbers = pd.DataFrame(), np.array([]), np.array([])
    y_train_full, y_train_pred_full = np.array([]), np.array([])
    cutting_train_size = False

    # Run the training-testing loop:
    for ii, (train_idx, test_idx) in tqdm(
        enumerate(time_splits.split(data)),
        total=time_splits.n_splits,
        disable=(not progress_bar) or verbose,
    ):
        if features:
            X_train = data[features].iloc[train_idx]
            X_test = data[features].iloc[test_idx]
        else:
            X_train = np.arange(len(data))[train_idx].reshape(-1, 1)
            X_test = np.arange(len(data))[test_idx].reshape(-1, 1)
        y_train = data[target].iloc[train_idx]
        y_test = data[target].iloc[test_idx]

        # Check that the iith time-split has as much training data as desired:
        if (
            train_idx.shape[0] != time_splits.max_train_size
            and cutting_train_size is False
        ):
            cutting_train_size = True
            print(
                "WARNING: there isn't enough data to fullfill the desired max_train_size for all time splits!"
            )

        # Create pipeline, optionally using StandardScaler (zero-mean-unit-variance):
        if use_scaler:
            model_pipeline = make_pipeline(StandardScaler(), clone(model))
        else:
            model_pipeline = clone(model)

        # Train model:
        model_pipeline.fit(X_train, y_train)

        # Predict:
        y_pred = model_pipeline.predict(X_test)
        # Predict also on training set:
        y_train_pred = model_pipeline.predict(X_train)

        # Evaluate this split:
        mae_test = mean_absolute_error(y_test, y_pred)
        rmse_test = mean_squared_error(y_test, y_pred, squared=False)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)

        # Print out some error metrics for each specific time-split if requested:
        if verbose:
            if time_splits.n_splits > 10:
                if (ii % 10 == 0) or (ii + 1 == time_splits.n_splits):
                    tqdm.write(
                        f"Train MAE|RMSE for fold {ii} is {mae_train:.2f} | {rmse_train:.2f} MW\n\
Test  MAE|RMSE for fold {ii} is {mae_test:.2f} | {rmse_test:.2f} MW"
                    )
            else:
                tqdm.write(
                    f"Train MAE|RMSE for fold {ii} is {mae_train:.2f} | {rmse_train:.2f} MW\n\
Test  MAE|RMSE for fold {ii} is {mae_test:.2f} | {rmse_test:.2f} MW"
                )

        # Save results:
        backtest = pd.concat([backtest, y_test], axis=0)
        predictions = np.append(predictions, y_pred)
        split_numbers = np.append(
            split_numbers, ii * np.ones(len(y_test), dtype=np.int16)
        )
        # Save also predictions on training data:
        y_train_full = np.append(y_train_full, y_train)
        y_train_pred_full = np.append(y_train_pred_full, y_train_pred)

    # Compile all numbers into a dataframe:
    backtest = backtest.rename(columns={0: "y_test"})
    backtest = backtest.assign(y_pred=predictions.astype(backtest["y_test"].dtype))
    backtest = backtest.assign(
        error=(backtest["y_test"] - backtest["y_pred"]).abs()
    ).assign(split_number=split_numbers.astype(np.int16))

    # And compute some error metrics for the entire forecasted time-series:
    total_mae = mean_absolute_error(backtest["y_test"], backtest["y_pred"])
    # Calculating MASE with respect to naive model:
    if "from_qh_plus_1" in target:
        diff_period = 2
    else:
        diff_period = 1
    if all_minutes:
        diff_period = 15 * diff_period
    total_mase = (
        total_mae / backtest["y_test"].diff(periods=diff_period).dropna().abs().mean()
    )
    #* Note that MASE should technically use the in-sample naïve MAE. In this particular methodology almost all of y_test is used to train models, so it is nearly exact when calculating over all of 2022.

    total_rmse = mean_squared_error(
        backtest["y_test"], backtest["y_pred"], squared=False
    )
    max_error = (backtest["y_test"] - backtest["y_pred"]).abs().max()
    max_error_dt = (backtest["y_test"] - backtest["y_pred"]).abs().idxmax()
    P90_error = backtest["error"].quantile(q=0.90)
    # Also compute a few for predictions on training set (to display possible overfitting):
    total_mae_train = mean_absolute_error(y_train_full, y_train_pred_full)
    total_rmse_train = mean_squared_error(
        y_train_full, y_train_pred_full, squared=False
    )

    end = perf_counter()
    if print_error_metrics:
        print(
            f"\nTrain set average error:\n\
MAE: {total_mae_train:.2f}MW | RMSE: {total_rmse_train:.2f}MW"
        )
        print(
            f"Test set average error:\n\
MAE: {total_mae:.2f}MW | RMSE: {total_rmse:.2f}MW | MASE: {total_mase:.4f} | \
P90 Error: {P90_error:.2f}MW | Max Error: {max_error:.1f} ({max_error_dt})"
        )
        print(
            f"Time elapsed: {end-start :.2f}s | Time per split: ~{(end-start)/time_splits.n_splits :.2f}s\n"
        )

    return [
        backtest,
        {
            "model": model,
            "tscv": time_splits,
            "MASE": total_mase,
            "MAE": total_mae,
            "RMSE": total_rmse,
            "max_error": max_error,
            "P90_error": P90_error,
        },
    ]


def main():
    pass


if __name__ == "__main__":
    main()
