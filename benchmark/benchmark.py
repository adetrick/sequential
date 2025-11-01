import pandas as pd
import numpy as np
import time
import random
import keras
import sequential
import json
import platform
import tensorflow as tf
from keras_util import *
from tqdm import tqdm

'''
Runs a benchmark between Keras and Sequential frameworks.

Tests two monthly files with basic configuration and simple
model architectures that stay fixed across files and both
frameworks.

Configuration
--------------

Dense, RNN, LSTM models:

Training data consists of overlapping sequences of 36 time
steps, each one predicting the next 12 time steps (i.e. the
models are learning to predict 1 year ahead given 3 years
of input data). Each model consists of 2 layers with 48 and
24 neurons, respectively, followed by a final Dense output
layer.

Transformers:

Training data consists of overlapping sequences of 60
time steps to predict the next 12 time steps (i.e. the
models are learning to predict 1 year ahead given 5 years
of input data). Each model consists of 2 decoder layers
with a model dimensionality of 50.

Training
--------

The benchmark iterates through 5 different random seeds
per dataset and reports the average validation loss and 
fit time (plus the standard deviation for each) across 
the runs.

All models were trained for a fixed number of N epochs
without early stopping to ensure identical training schedules 
across frameworks.


Validation
----------

Validation loss is computed after training using autoregressive 
forecasts.

A validation set of 12 time steps is separated from each dataset
before creating the training sequences. After fitting the
model, an autoregressive forecast is made and the difference
between the forecast and real data is logged as `val_loss`.

Autoregressive forecasting means that for each step in a 
forecast of length n, a full sequence of `time_steps` is
predicted but only the first value is kept as the forecsted 
value for future time step t. The forecasted value is prepended 
to the input sequence and the cycle continues until the forecast 
steps are complete.

Sequential generates autoregressive forecasts natively with 
the get_forecast() method. However, Keras does not so the 
generate_keras_forecast() function was created.


Data sources
------------

AirPassengers: https://www.kaggle.com/datasets/rakannimer/air-passengers

useia_renewable_energy_production.csv: https://www.eia.gov/renewable/data.php,
downloaded from https://www.datamago.com/open-data/renewable-energy-consumption
'''


def run_benchmark():

    benchmark_start = time.perf_counter()

    # force deterministic kernals for reproducibility
    tf.config.experimental.enable_op_determinism()

    # model configuration
    config = {
        'time_steps': 36,
        'target_time_steps': 12,
        'learning_rate': 1e-3,
        'loss': "mae",
        'epochs': 10,
    }

    # transformer specific configs
    transformer_config = {
        'num_layers': 2,
        'd_model': 50,
        'units': 64,
        'time_steps': 60,
        'num_heads': 1,
    }

    # model architectures for each model type
    model_map = {
        'dense': {
            'sequential': [
                sequential.layers.Dense(48, activation="relu"),
                sequential.layers.Dense(24, activation='relu'),
                sequential.layers.Dense(1, activation=None),
            ],
            'keras': [
                keras.layers.Flatten(),
                keras.layers.Dense(48, activation="relu"),
                keras.layers.Dense(24, activation="relu"),
                keras.layers.Dense(config['target_time_steps']),
                keras.layers.Reshape((config['target_time_steps'], 1))
            ],
        },
        'rnn': {
            'sequential': [
                sequential.layers.RNN(48),
                sequential.layers.RNN(24),
                sequential.layers.Dense(1, activation=None)
            ],
            'keras': [
                keras.layers.SimpleRNN(48, return_sequences=True),
                keras.layers.SimpleRNN(24, return_sequences=False),
                keras.layers.Dense(config['target_time_steps']),
                keras.layers.Reshape((config['target_time_steps'], 1))
            ]
        },
        'lstm': {
            'sequential': [
                sequential.layers.LSTM(48),
                sequential.layers.LSTM(24),
                sequential.layers.Dense(1, activation=None),
            ],
            'keras': [
                keras.layers.LSTM(48, return_sequences=True),
                keras.layers.LSTM(24, return_sequences=False),
                keras.layers.Dense(config['target_time_steps']),
                keras.layers.Reshape((config['target_time_steps'], 1))
            ]
        },
        'transformer': {
            'sequential': transformer_config,
            'keras': transformer_config
        }
    }

    files = ['../data/AirPassengers.csv', '../data/useia_renewable_energy_production.csv']
    results = {}

    for file_path in files:
        filename = file_path.split('/')[-1]
        print(f"\n=== {filename} ===\n")
        # load data into a dataframe
        df = pd.read_csv(file_path)
        df.drop(columns=['Date'], inplace=True)
        # convert to numpy array
        X_orig = df.to_numpy()

        # use target time steps as the validation set length
        val_len = config['target_time_steps']

        # scale and turn X into sequences for sequential models
        X, X_train, X_test, y_train, y_test, scaler = sequential.preprocessing.preprocessing(
            X_orig, val_len, config['time_steps'], target_time_steps=config['target_time_steps'])

        # scale and turn X into sequences for transformer models
        X_transformer, X_train_transformer, X_test_transformer, y_train_transformer, y_test_transformer, scaler_transformer = sequential.preprocessing.preprocessing(
            X_orig, val_len, transformer_config['time_steps'], target_time_steps=config['target_time_steps'], autoregressive=True)

        # save a dictionary of results for each random seed
        results[filename] = {
            'sequential': {k: [] for k in model_map.keys()},
            'keras': {k: [] for k in model_map.keys()}
        }

        for seed in tqdm([3, 7, 100, 500, 1000], desc=f"{filename} seeds"):

            random.seed(seed)
            np.random.seed(seed)
            keras.utils.set_random_seed(seed)

            for model_type in model_map:

                for framework, model_arch in model_map[model_type].items():

                    print(f"{framework.title()} {model_type} model with random seed {seed}")

                    X_train_to_use = X_train
                    X_test_to_use = X_test
                    y_train_to_use = y_train
                    y_test_to_use = y_test
                    if model_type == 'transformer':
                        X_train_to_use = X_train_transformer
                        X_test_to_use = X_test_transformer
                        y_train_to_use = y_train_transformer
                        y_test_to_use = y_test_transformer

                    if framework == 'sequential':
                        if model_type == 'transformer':
                            model = sequential.models.Transformer(
                                num_decoder_layers=model_arch['num_layers'],
                                d_model=model_arch['d_model'],
                                units=model_arch['units'],
                                num_heads=model_arch['num_heads'],
                                optimizer_args={'alpha': config['learning_rate']},
                                loss=config['loss'])
                        else:
                            model = sequential.models.NeuralNet(
                                model_arch, optimizer='adam', optimizer_args={'alpha': config['learning_rate']}, loss=config['loss'])

                        fit_kwargs = {
                            'epochs': config['epochs'],
                            'verbose_rate': 0,
                        }

                    else:
                        if model_type == 'transformer':
                            model = build_keras_decoder_only_transformer(
                                input_shape=(X_train_to_use.shape[1], X_train_to_use.shape[2]),
                                target_time_steps=config['target_time_steps'],
                                d_model=model_arch['d_model'],
                                num_heads=model_arch['num_heads'],
                                num_layers=model_arch['num_layers'],
                                ff_units=model_arch['units']
                            )
                        else:
                            model = keras.Sequential(model_arch)

                        model.compile(loss=config['loss'], optimizer=keras.optimizers.Adam(
                            learning_rate=config['learning_rate']))

                        fit_kwargs = {
                            'epochs': config['epochs'],
                            'verbose': 0,
                            'shuffle': False,
                        }

                    fit_start = time.perf_counter()

                    if framework == 'sequential':
                        fitted_values, loss_history = model.fit(
                            X_train_to_use, y_train_to_use, **fit_kwargs)
                    else:
                        history = model.fit(X_train_to_use, y_train_to_use, **fit_kwargs)
                        loss_history = history.history['loss']

                    fit_time = time.perf_counter() - fit_start

                    if framework == 'sequential':
                        forecast = model.get_forecast(
                            config['target_time_steps'], init_input=X_test_to_use)
                    else:
                        forecast = generate_keras_forecast(
                            config['target_time_steps'], X_test_to_use, model)

                    results[filename][framework][model_type].append({
                        'fit_time': fit_time,
                        'fit_loss': np.round(loss_history[-1], 6),
                        'val_loss': np.round(sequential.metrics.mean_squared_error(
                            y_test_to_use, forecast), 6)
                    })

    run_time = (time.perf_counter() - benchmark_start) / 60

    # calculate the val loss and fit time mean/std across
    # files and random seeds
    summary = {
        # save environment info for reproducibility
        'environment': {
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "numpy_version": np.__version__,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        'run_time_minutes': run_time,
        # track global mean loss/fit time per framework
        'summary': {},
    }

    for file in files:
        filename = file.split('/')[-1]
        summary[filename] = {
            'sequential': {k: {} for k in model_map.keys()},
            'keras': {k: {} for k in model_map.keys()}
        }
        for framework, models_res in results[filename].items():
            # track loss and fit times across all files and seeds
            # for each framework
            if framework not in summary['summary']:
                summary['summary'][framework] = {
                    'val_losses': [],
                    'fit_losses': [],
                    'fit_times': [],
                }
            for model_type in models_res:
                val_losses = [r["val_loss"] for r in models_res[model_type]]
                fit_losses = [r["fit_loss"] for r in models_res[model_type]]
                fit_times = [r["fit_time"] for r in models_res[model_type]]
                summary[filename][framework][model_type] = {
                    'val_loss_mean': np.mean(val_losses),
                    'val_loss_std': np.std(val_losses),
                    'fit_loss_mean': np.mean(fit_losses),
                    'fit_loss_std': np.std(val_losses),
                    'fit_time_mean': np.mean(fit_times),
                    'fit_time_std': np.std(fit_times)
                }
                # add results to global val losses and fit times
                summary['summary'][framework]['val_losses'] += val_losses
                summary['summary'][framework]['fit_losses'] += fit_losses
                summary['summary'][framework]['fit_times'] += fit_times

    # calculate the mean and std for global losses and fit times
    for framework, data in summary['summary'].items():

        summary['summary'][framework]['val_loss_mean'] = np.mean(data['val_losses'])
        summary['summary'][framework]['val_loss_std'] = np.std(data['val_losses'])
        summary['summary'][framework]['fit_loss_mean'] = np.mean(data['fit_losses'])
        summary['summary'][framework]['fit_loss_std'] = np.std(data['fit_losses'])
        summary['summary'][framework]['fit_time_mean'] = np.mean(data['fit_times'])
        summary['summary'][framework]['fit_time_std'] = np.std(data['fit_times'])
        # clean up
        del summary['summary'][framework]['val_losses']
        del summary['summary'][framework]['fit_losses']
        del summary['summary'][framework]['fit_times']

    print("\n=== Benchmark Summary ===")
    print(f"Total run time: {run_time:.2f} minutes")

    print('\nTotal mean validation loss:')
    for framework in ['sequential', 'keras']:
        print(
            f"{framework.title()}: {summary['summary'][framework]['val_loss_mean']:.4f} ({summary['summary'][framework]['val_loss_std']:.2f} std)")

    print('\nTotal mean fit loss:')
    for framework in ['sequential', 'keras']:
        print(
            f"{framework.title()}: {summary['summary'][framework]['fit_loss_mean']:.4f} ({summary['summary'][framework]['fit_loss_std']:.2f} std)")

    print('\nTotal mean fit time:')
    for framework in ['sequential', 'keras']:
        print(
            f"{framework.title()}: {summary['summary'][framework]['fit_time_mean']:.2f} ({summary['summary'][framework]['fit_time_std']:.2f} std)")

    print("\n=== Results per file ===")

    # save flattened results for CSV file
    flat_records = []

    for filename, data in summary.items():
        if '.csv' not in filename:
            continue
        print(f"\n--- {filename} ---")
        for framework, models in data.items():
            print(f"\n-- {framework.title()} --")
            for model_type, stats in models.items():
                print(f"{model_type.title()}:")
                print(f"Val Loss:  {stats['val_loss_mean']:.4f} ± {stats['val_loss_std']:.4f}")
                print(f"Fit Loss:  {stats['fit_loss_mean']:.2f}s ± {stats['fit_loss_std']:.2f}s")
                print(f"Fit Time:  {stats['fit_time_mean']:.2f}s ± {stats['fit_time_std']:.2f}s")
                flat_records.append({
                    "dataset": filename,
                    "framework": framework,
                    "model_type": model_type,
                    **stats
                })

    # save results as JSON
    with open("benchmark-res.json", "w") as json_file:
        json.dump(summary, json_file, indent=4)

    # save results as CSV
    pd.DataFrame(flat_records).to_csv("benchmark-res.csv", index=False)


if __name__ == "__main__":
    run_benchmark()
