import os
import argparse
import json
import pickle
import time
import pandas as pd
import numpy as np
from sequential.models import NeuralNet, Transformer
from sequential.layers import Dense, LSTM, RNN, Dropout, LayerNorm
from sequential.preprocessing import *
from sequential.metrics import mean_absolute_error
from utils import *
import matplotlib.pyplot as plt


def run(
        path,
        time_steps,
        val_len,
        target_time_steps=None,
        epochs=100,
        batch_size=None,
        loss="mse",
        optimizer="adam",
        optimizer_args=None,
        verbose_rate=10,
        val_features_path=None,
        forecast_features_path=None,
        model_type="sequential",
        random_seed=None,
        **kwargs):
    '''
    Run a model training and forecasting pipeline for sequential data. 
    Includes model fitting, forecasting, and saving plots and CSV files 
    of the errors, fitted values, validation forecast, and final forecast.

    Supports both sequential models (Dense, RNN, LSTM) and Transformer models.

    Args
    ----
    path: str
        Path to a CSV file containing the training data. 
        If multiple columns are present, the first column is 
        assumed to be the target. If a 'date' column exists and 
        is successfully parsed, future dates will be added to 
        forecast output files.
    time_steps: int
        Number of time steps in each training sequence.
    val_len: int
        Number of rows to reserve for validation.
    target_time_steps: int, optional
        Number of time steps in each target sequence, will equal 
        time_steps if not specified.
    epochs: int, optional
        Number of training iterations.
    batch_size: int, optional
        Number of samples per update step. If None, uses full batch.
    loss: str, optional
        Loss function to minimize. Options: 'mse', 'mae'.
    optimizer: str, optional
        Optimization algorithm for updating parameters. Options: 
        'gd' (gradient descent), 'adam'.
    optimizer_args: dict, optional
        Dictionary of hyperparameters passed to the optimizer (e.g., {'alpha': 0.001}). 
    verbose_rate: int, optional
        Frequency (in epochs) for printing the training loss. For example, a value 
        of 10 prints the loss every 10 epochs. Set to 0 to disable. Training 
        is usually slightly faster without verbosity.
    val_features_path: str, required if inputs contain features
        Path to a CSV file containing feature values for generating the validation 
        forecast. Number of features must equal the number of features in the 
        input data, with the number of rows equaling `val_len`.
    forecast_features_path: str, required if inputs contain features
        Path to a CSV file containing future features values for generating the 
        forecast. Number of features must equal the number of features in the 
        input data, with the number of rows equaling `target_time_steps`, or 
        `time_steps` if the former isn't specified.
    model_type: str, optional
        Type of model to run. Options: sequential, transformer.
    random_seed: int, optional
        Random seed for NumPy's random number generator. Ensures reproducible
        model fitting across runs. If None, no seed is set and results may
        vary between runs.
    kwargs:
        Additional model-specific arguments.
        - For `sequential` model type: see class `sequential.models.NeuralNet`.
        - For `transformer` model type: see class `sequential.models.Transformer`.

    Command line usage
    ------------------
    When running this script from the command line, parameters are placed in 
    a JSON file. Each top-level JSON field corresponds to a parameter in run(). 
    Any additional fields are passed directly as **kwargs to the model constructor. 
    In the case of `sequential` models, each layer in the `layers` parameter 
    must be a dictionary with a "class" field specifying its type. This will 
    automatically be converted to a list of instantiated layer objects before
    passing it to the NeuralNet constructor.

    Config examples: note that LSTM, RNN, Dense, LayerNorm, and Dropout layers
    can be mixed and that an output layer containing 1 unit is automatically
    added to the layers list if not included in the config file.

        `sequential` model type with Dense layers:
            {
                "path": "../data/AirPassengers.csv",
                "model_type": "sequential",
                "layers": [
                    {"class": "Dense", "units": 48, "activation": "relu", "relu_alpha": 0.01},
                    {"class": "Dense", "units": 12, "activation": "relu", "relu_alpha": 0.01}
                ],
                "time_steps": 36,
                "target_time_steps": 12,
                "val_len": 12,
                "epochs": 1000,
                "loss": "mae",
                "optimizer": "adam",
                "optimizer_args": {
                    "alpha": 0.001
                }
            }

        `sequential` model type with RNN layers:
            {
                "path": "../data/AirPassengers.csv",
                "model_type": "sequential",
                "layers": [
                    {"class": "RNN", "units": 64, "stateful": false},
                    {"class": "RNN", "units": 48, "stateful": false},
                ],
                "time_steps": 36,
                "target_time_steps": 12,
                "val_len": 12,
                "epochs": 2500,
                "loss": "mae",
                "optimizer": "adam",
                "optimizer_args": {
                    "alpha": 0.001
                }
            }

        `sequential` model type with LSTM and Dense layers: 
            {
                "path": "../data/AirPassengers.csv",
                "model_type": "sequential",
                "layers": [
                    {"class": "LSTM", "units": 54, "stateful": true},
                    {"class": "Dense", "units": 18, "activation": "relu", "relu_alpha": 0.01}
                ],
                "time_steps": 36,
                "target_time_steps": 12,
                "val_len": 12,
                "epochs": 2000,
                "loss": "mae",
                "optimizer": "adam",
                "optimizer_args": {
                    "alpha": 0.001
                }
            }

        `transformer` model type:
            {
                "path": "../data/AirPassengers.csv",
                "model_type": "transformer",
                "d_model": 50,
                "num_heads": 1,
                "num_decoder_layers": 6,
                "units": 64,
                "normalize": false,
                "drop_rate": 0,
                "time_steps": 120,
                "target_time_steps": 12,
                "val_len": 12,
                "epochs": 1000,
                "loss": "mae",
                "optimizer": "adam",
                "optimizer_args": {
                    "alpha": 0.001
                }
            }
    '''

    start_time = time.perf_counter()
    model_id = int(time.time())

    # set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # set target time steps to equal time steps if not provided
    if target_time_steps is None:
        target_time_steps = time_steps

    validate_args(path, model_type, time_steps, target_time_steps,
                  val_features_path, forecast_features_path)

    df, dates, date_freq = load_data(path)

    # use the numpy representation of the inputs for processing/training
    X = df.to_numpy()

    X, X_train, X_test, y_train, y_val, scaler = preprocessing(
        X, val_len, time_steps, target_time_steps=target_time_steps, autoregressive=model_type == 'transformer')

    forecast_features, val_features = load_features(
        X, forecast_features_path, val_features_path, val_len, target_time_steps)

    # ---- model fit ----

    if model_type == "sequential":
        # append output layer if not included
        if kwargs["layers"][-1].units > 1:
            kwargs["layers"].append(Dense(1, activation=None))

        # instantiate sequential model (Dense, RNN, LSTM)
        model = NeuralNet(**kwargs)

    elif model_type == "transformer":

        # instantiate transformer model
        model = Transformer(**kwargs)

    # fit the model to the training data
    fitted_val, errors = model.fit(X_train, y_train, epochs=epochs,
                                   batch_size=batch_size, verbose_rate=verbose_rate)

    # calc the mean absolute error for the fitted values
    mae = mean_absolute_error(y_train.flatten(), fitted_val.flatten())

    # ---- forecasts ----

    if val_features is not None:
        val_features = scale_features(val_features, scaler)

    # generate validation forecast
    val_forecast = model.get_forecast(val_len, init_input=X_test, features=val_features)

    # calc the mean absolute error for the validation forecast
    val_mae = mean_absolute_error(y_val, val_forecast)

    # forecast into the future using the last sequence of X as
    # the input
    init_input = X[-1][np.newaxis, :, :]

    if forecast_features is not None:
        forecast_features = scale_features(forecast_features, scaler)

    # generate forecast
    forecast = model.get_forecast(
        target_time_steps, init_input=init_input, features=forecast_features)

    # reverse scaling on forecast
    forecast = inverse_target_scaling(forecast, scaler)

    # ---- save results ----

    if not os.path.exists('model_fit/'):
        os.mkdir('model_fit/')

    # create a directory name for saving current model fit data
    dirname = f"model_fit/{model_type}_model_{model_id}"

    # create directory
    os.mkdir(dirname)

    # fitted values csv
    pd.DataFrame({"Pred": fitted_val.flatten(), "Actual": y_train.flatten()}
                 ).to_csv(f"{dirname}/training_all.csv")

    # fitted values plot
    save_plot([y_train, fitted_val],
              f"Training (all windows), MAE: {mae}", f"{dirname}/training_all.png")

    # csv and plot of fitted values for each sequence
    num_seqs = y_train.shape[0]
    for i in range(target_time_steps):
        step = i + 1
        fit_path = f"{dirname}/training_{step}_of_{target_time_steps}"
        dates_i = dates[i: i + num_seqs] if dates is not None else None
        fitted_val_i = fitted_val[:, i, :].flatten()
        y_train_i = y_train[:, i, :].flatten()
        # csv
        pd.DataFrame({"Pred": fitted_val_i, "Actual": y_train_i},
                     index=dates_i).to_csv(fit_path + ".csv")
        # plot
        mae_i = mean_absolute_error(y_train_i, fitted_val_i)
        save_plot([y_train_i, fitted_val_i],
                  f"Training ({step} of {target_time_steps}), MAE: {mae_i}", fit_path + ".png")

    # validation forecast plot
    save_plot([y_val, val_forecast],
              f"Validation forecast, MAE: {val_mae}", f"{dirname}/validation_forecast.png")

    # validation forecast csv
    pd.DataFrame({"Val forecast": val_forecast.flatten(), "Actual": y_val.flatten()}
                 ).to_csv(f"{dirname}/validation_forecast.csv")

    # error per epoch plot
    save_plot([errors], f"Error per epoch", f"{dirname}/errors.png")

    # error per epoch csv
    epoch_df = pd.DataFrame(errors, columns=['Error'], index=np.arange(len(errors)) + 1)
    epoch_df.index.name = "Epoch"
    epoch_df.to_csv(f"{dirname}/errors.csv")

    # forecast csv
    forecast_len = forecast.shape[0]
    forecast_index = np.arange(df.shape[0] + forecast_len)
    if dates is not None and date_freq is not None:
        fdates = pd.Series(pd.date_range(
            start=dates.iloc[-1], periods=forecast_len + 1, freq=date_freq)[1:])
        forecast_index = pd.concat([dates, fdates])
    forecast_df = pd.DataFrame(index=forecast_index, columns=['Actual', 'Forecast'])
    # save the historical target values
    forecast_df.iloc[:-forecast_len, 0] = df.iloc[:, 0]
    # save the forecast
    forecast_df.iloc[-forecast_len:, 1] = forecast.flatten()
    forecast_df.to_csv(f"{dirname}/forecast.csv")

    # forecast plot
    forecast_plot = np.append(np.full(df.shape[0], np.inf), forecast)
    save_plot([df.iloc[:, 0], forecast_plot],
              f"Forecast (original scale)", f"{dirname}/forecast.png")

    # save model as a pickle file
    with open(f"{dirname}/model.pkl", "wb") as file:
        pickle.dump(model, file)

    # store config options in a dictionary
    config = {
        "model_type": model_type,
        "path": path,
        "time_steps": time_steps,
        "val_len": val_len,
        "target_time_steps": target_time_steps,
        "epochs": epochs,
        "batch_size": batch_size,
        "loss": loss,
        "optimizer": optimizer,
        "optimizer_args": optimizer_args,
    }

    if val_features_path:
        config["val_features_path"] = val_features_path

    if forecast_features_path:
        config["forecast_features_path"] = forecast_features_path

    # merge config and model specific kwargs
    config = config | kwargs

    # convert list of layer objects to a list of dictionaries
    # for JSON serialization
    if model_type == "sequential" and config["layers"]:
        config["layers"] = createLayersDictList(config["layers"])

    # save config to json
    with open(f"{dirname}/config.json", "w") as file:
        json.dump(config, file, indent=4)

    print(f"Fit MAE: {mae}")
    print(f"Validation MAE: {val_mae}")
    print(f"Run time: {time.perf_counter() - start_time:.2f}s")

    return model_id, mae, val_mae


def save_plot(series_list, title, path):
    for s in series_list:
        plt.plot(s.flatten() if hasattr(s, "flatten") else s)
    plt.title(title)
    # save the plot to a file
    plt.savefig(path)
    plt.close()


def createLayersDictList(layers):
    '''
    creates a serializable list of layer configs from 
    a list of layer objects
    '''
    layers_list = []
    # attributes to save
    attrs = ['activation', 'use_bias', 'units', 'drop_rate', 'stateful', 'relu_alpha']
    for layer in layers:
        layer_dict = {"class": type(layer).__name__}
        # iterate through the attributes, copying those
        # that the layer contains
        for attr in attrs:
            if hasattr(layer, attr):
                layer_dict[attr] = layer.__getattribute__(attr)

        layers_list.append(layer_dict)

    return layers_list


def createLayerObjects(layers_list):
    '''
    layers_list: list of dicts
        dict properties:
            {
                'class': str, required
                    options: 'dense', 'rnn', 'lstm', 'dropout', 'norm'
                'units': int, required
                    number of neurons
                'use_bias': bool, optional
                'activation': str, optional
                    options: 'relu', 'tanh'
                'relu_alpha': float, optional
                    prevents vanishing gradients
                'stateful': bool, optional
                    preserves hidden states between batches/epochs, 
                    only applies to lstm or rnn layers
            }
    '''
    layers = []
    for i, layer_conf in enumerate(layers_list):
        layer_class = None
        classname = layer_conf['class'].lower()
        if classname == 'dense':
            layer_class = Dense
        elif classname == 'rnn':
            layer_class = RNN
        elif classname == 'lstm':
            layer_class = LSTM
        elif classname == 'dropout':
            layer_class = Dropout
        elif classname == 'norm':
            layer_class = LayerNorm

        if layer_class is None:
            raise ValueError(
                f"Layer class '{layer_conf['class']}' not found, supported options include: 'Dense', 'RNN', 'LSTM', or 'Dropout'")

        # remove 'class' to prevent unexpected keyword error
        del layer_conf['class']

        layer_obj = layer_class(**layer_conf)

        if i == 0 and not isinstance(layer_obj, (Dense, RNN, LSTM)):
            raise ValueError("The first layer must be a Dense, RNN, or LSTM layer")

        layers.append(layer_obj)

    return layers


def validate_args(path, model_type, time_steps, target_time_steps, val_features_path, forecast_features_path):
    if not os.path.exists(path):
        raise ValueError(f"Invalid path: {path}")
    if model_type not in ["sequential", "transformer"]:
        raise ValueError(f"model_type must be `sequential` or `transformer`")
    if not time_steps:
        raise ValueError("must provide time_steps")
    if int(time_steps) < 1:
        raise ValueError("time_steps must be 1 or above")
    if target_time_steps is not None:
        if int(target_time_steps) < 1:
            raise ValueError("target_time_steps must be 1 or above")
        if int(target_time_steps) > int(time_steps):
            raise ValueError("target_time_steps must be <= time_steps")
    for p in [val_features_path, forecast_features_path]:
        if p and not os.path.exists(p):
            raise ValueError(f"Invalid path: {p}")


def load_features(X, forecast_features_path, val_features_path, val_len, target_time_steps):

    num_features = X.shape[-1]
    forecast_features = None
    val_features = None

    if num_features > 1:
        # load features into numpy arrays
        for path in [forecast_features_path, val_features_path]:
            features = load_data(path) if path else None
            features = features.to_numpy() if features is not None else None
            if path == forecast_features_path:
                forecast_features = features
            else:
                val_features = features

        validate_features(num_features, forecast_features, val_features, val_len, target_time_steps)

    return forecast_features, val_features


def validate_features(num_features, forecast_features, val_features, val_len, target_time_steps):

    if forecast_features is None or val_features is None:
        raise ValueError(
            "Both forecast features and val_features are required when the training inputs (X) contains features")

    if forecast_features.shape[-1] != num_features or val_features.shape[-1] != num_features:
        raise ValueError(
            "Both forecast and validation forecast feature values must contain the same number of columns as the training data")

    if val_features.shape[-1] != val_len:
        raise ValueError(
            "The number of rows in the validation forecast features must match val_len (validation set length)")

    if forecast_features.shape[-1] != target_time_steps:
        raise ValueError(
            "The number of rows in the forecast features must math the target time steps, or time steps if the former isn't specified")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        "-cp",
        type=str,
        nargs="+",
        help="path to config json file"
    )

    args = parser.parse_args()

    # default configs
    def_config = {
        "path": "../data/AirPassengers.csv",
        "layers": [Dense(24), Dense(12)],
        "time_steps": 24,
        "target_time_steps": 12,
        "val_len": 12,
        "epochs": 1000,
        "batch_size": None,
        "loss": "mse",
        "optimizer": "adam",
        "optimizer_args": {"alpha": .001},
        "model_type": "sequential",
    }

    # load user defined configs
    if args.config:
        # in case multiple configs are provided, keep track of
        # which model produces the lowest fitted and validation mae
        best_mae = {"model_id": None, "mae": np.inf}
        best_val_mae = {"model_id": None, "mae": np.inf}
        start_time = time.perf_counter()

        # iterate through the config files
        for arg_config in args.config:
            with open(arg_config, "r") as f:
                config_data = json.load(f)

            # turn layers config into list of layer objects
            if config_data.get("layers") and config_data["model_type"] == "sequential":
                config_data["layers"] = createLayerObjects(config_data["layers"])

            # merge default and user provided configs
            config = def_config | config_data

            if config["model_type"] == "transformer" and config.get("layers"):
                del config["layers"]

            model_id, mae, val_mae = run(**config)

            if mae < best_mae["mae"]:
                best_mae = {"model_id": model_id, "mae": mae}

            if val_mae < best_val_mae["mae"]:
                best_val_mae = {"model_id": model_id, "mae": val_mae}

        if len(args.config) > 1:
            print("------------")
            print(f"Lowest fit MAE: {best_mae["mae"]} from model {best_mae["model_id"]}")
            print(
                f"Lowest validation MAE: {best_val_mae["mae"]} from model {best_val_mae["model_id"]}")
            print(f"Total run time: {time.perf_counter() - start_time:.2f}s")

    else:
        run(**def_config)
