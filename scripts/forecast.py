import os
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from sequential.preprocessing import MinMaxScaler, create_sequences


def run(
        path,
        model_path,
        time_steps,
        features_path=None):
    '''
    Run a forecasting pipeline using a previously trained model. 

    Loads a saved model from disk, generates a forecast for the 
    specified number of future time steps, and saves both a plot 
    and a CSV file of the results.

    Args
    ----
    path: str
        Path to a CSV file containing the historical values 
        that will serve as as starting point for the forecast.
        If multiple columns are present, the first column is 
        assumed to be the target. If a 'date' column exists and 
        is successfully parsed, future dates will be added to 
        forecast output files.
    model_path: str
        Path to a trained model pickle file.
    time_steps: int
        Number of time steps to forecast into the future.
    features_path: str
        Path to a CSV file containing future feature values, with 
        one row per forecast step. Only applicable if the model 
        was trained with exogenous features.
    '''

    validate_args(path, model_path, time_steps, features_path)

    # load trained model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    df, dates, date_freq = load_data(path)

    scaler = MinMaxScaler()

    # scale historical data to (0, 1)
    X = scaler.fit_transform(df.values)

    # get the time steps from the training data for
    # creating sequences
    training_time_steps = model.X.shape[1]

    # convert last sequence of historical data to 3D: (1, time_steps, features)
    # for initializing the forecast
    X = create_sequences(X[-training_time_steps:], training_time_steps)

    # load future features
    forecast_features = None
    if features_path is not None:
        forecast_features = pd.read_csv(features_path).to_numpy()

    validate_features(X, forecast_features)

    # generate forecast
    if forecast_features is not None:
        forecast_features = scale_features(forecast_features, scaler)

    forecast = model.get_forecast(time_steps, init_input=X, features=forecast_features)

    # invert scaling
    forecast = inverse_target_scaling(forecast, scaler)

    # ---- save results ----

    # save the forecast plot and CSV to the same folder as the model
    dirpath = '.'
    if '/' in model_path:
        dirpath = '/'.join(model_path.split('/')[:-1])

        # forecast csv
    forecast_index = np.arange(df.shape[0] + time_steps)
    if dates is not None and date_freq is not None:
        fdates = pd.Series(pd.date_range(
            start=dates.iloc[-1], periods=time_steps + 1, freq=date_freq)[1:])
        forecast_index = pd.concat([dates, fdates])
    forecast_df = pd.DataFrame(index=forecast_index, columns=['Actual', 'Forecast'])
    # save the historical target values
    forecast_df.iloc[:-time_steps, 0] = df.iloc[:, 0]
    # save the forecast
    forecast_df.iloc[-time_steps:, 1] = forecast.flatten()
    forecast_df.to_csv(f"{dirpath}/forecast_{time_steps}_steps.csv")

    # forecast plot
    forecast_plot = np.append(np.full(df.shape[0], np.inf), forecast)
    save_plot([df.iloc[:, 0], forecast_plot], f"Forecast",
              f"{dirpath}/forecast_{time_steps}_steps.png")

    print(f"Forecast complete, results saved to `{dirpath}/`")

    return forecast


def save_plot(series_list, title, path):
    for s in series_list:
        plt.plot(s.flatten() if hasattr(s, "flatten") else s)
    plt.title(title)
    # save the plot to a file
    plt.savefig(path)
    plt.close()


def validate_args(path, model_path, time_steps, features_path):
    if not os.path.exists(path):
        raise ValueError(f"Invalid path to historical data: {path}")

    if not os.path.exists(model_path):
        raise ValueError(f"Invalid path to trained model: {model_path}")

    assert time_steps > 0

    if features_path is not None and not os.path.exists(features_path):
        raise ValueError(f"Invalid path to future feature values: {features_path}")


def validate_features(X, features):
    num_features = X.shape[-1]
    if num_features > 1:
        if features is None:
            raise ValueError(
                f"future feature values must be provided as the model was trained with {num_features}")
        if features.shape[-1] != num_features:
            raise ValueError(
                f"the number of features provided ({features.shape[1]}) must match the number of features in the training data ({num_features})")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "path",
        type=str,
        help="path to historical data to forecast"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="path to trained model pickle file"
    )
    parser.add_argument(
        "time_steps",
        type=int,
        help="number of time steps to forecast into the future"
    )
    parser.add_argument(
        "--features_path",
        "-f",
        type=str,
        default=None,
        help="path to file with future feature values for generating the forecast. Only applicable if the model was trained with multiple features."
    )

    args = parser.parse_args()

    run(args.path, args.model_path, args.time_steps, features_path=args.features_path)
