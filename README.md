# Sequential

Sequential is a fast, lightweight framework for multi-step forecasting of sequential data with neural networks. It supports feed-forward, recurrent (RNN and LSTM), and decoder-only Transformer models, implemented in pure Python and NumPy.

It handles both many-to-one and many-to-many tasks, where the model predicts a sequence of length `n` for each input. `n` can be any length up to the input sequence length, and Sequential takes care of any differences automatically.

Built with **explorability and learning in mind**, its limited modules and clear docstrings make it easy to audit, extend, and understand how neural networks function under the hood. Unlike larger frameworks such as Keras, TensorFlow, and PyTorch, which offer extensive, feature-rich APIs across many domains, Sequential focuses on **time-series forecasting**, delivering a simpler yet highly capable interface for deep learning.

Benchmarking and a blog series detailing the framework’s development are coming soon.

## Installation

### Base install

Sequential requires Python >= 3.12 and NumPy >= 2.0.0. To install, clone the project and run:

```bash
pip install .
```


### Optional dependencies

CLI tools & Jupyter notebooks (adds Pandas, Matplotlib):

```bash
pip install '.[cli]'
```

Development & testing (adds Pytest):

```bash
pip install '.[dev]'
```

## Usage

Sequential can be used from the command line or imported into custom scripts. 

### Command line 

Two command line tools are available in the `sequential/scripts/` directory:

#### train.py

This script provides an automated pipeline for training models, including preprocessing, fitting, validating, forecasting, and outputting CSVs and plots (for the loss, fitted values, and forecasts). It supports feed-forward, recurrent (RNN, LSTM), and Transformer models.

Parameters are passed to its `run()` function via a JSON file. The terminal command looks like this:

```bash
python train.py --config path_to_json_file
```

The JSON file should contain the required `run()` parameters you see in `train.py` (e.g. path to the training data, time steps, validation length), plus any defaults you want to override and model-specific parameters handled by `**kwargs`.

Example JSON configuration for an RNN model:

```json
    {
        "path": "path_to_training_data",
        "model_type": "sequential",
        "layers": [
            {"class": "RNN", "units": 64, "stateful": false},
            {"class": "RNN", "units": 12, "stateful": false}
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
```

There are two example JSON files in the `sequential/scripts/` directory (`sequential-config-example.json` and `transformer-config-example.json`) to help you get started, and there are more details about accepted parameters and example configs documented in `train.py`'s docstring.

After fitting, CSV files and plots (loss, fitted values, and forecasts) are saved to `model_fit/`, which is created in the working directory.

(Requires `.[cli]` extras; see Installation section.)

#### forecast.py

After running `train.py`, the trained model is saved to a pickle file within the `model_fit/` directory, which will be created if not present. This script processes the historical data, loads the saved model, and generates a forecast for the desired number of time steps. 

Basic usage:

```bash
python forecast.py path_to_historical_data path_to_pickle_file time_steps 
```

For a full list of options and arguments, run:

```bash
python3 forecast.py -h
```

### Jupyter notebooks

There are two jupyter notebook templates in the `sequential/scripts/` directory, one for feed-forward and recurrent networks (`sequential.ipynb`) and the other for Transformers (`transformer.ipynb`). Both contain code to load historical data, run preprocessing, fit models, perform validation, and forecast beyond the historical data.

(Requires `.[cli]` extras; see Installation section.)

### Custom scripts

After installing Sequential, regular imports can be used to fit models and generate predictions in your own scripts. In addition to model instantiation, fitting, and forecasting, Sequential includes preprocessing functions to simplify data preparation.

#### Feed-forward and recurrent models

Here's an example of how you could create an LSTM model with monthly time series data:

```python
import numpy as np
from sequential.models import NeuralNet
from sequential.preprocessing import preprocessing
from sequential.layers import *

'''
Load training data here. The input data X should be shaped (num_samples, num_features).
'''

# number of time steps in each sequence
time_steps = 60 
# number of rows to set aside for validation
val_len = 12
# number of time steps in each target sequence, must 
# be <= time_steps
target_time_steps = 12

# preprocessing scales X to (0, 1), converts it into overlapping 
# 3D sequences (num_sequences, time_steps, num_features), and 
# splits it into training and validation sets.
X, X_train, X_test, y_train, y_val, scaler = preprocessing(
        X, val_len, time_steps, target_time_steps=target_time_steps, autoregressive=False)

# create a list of instantiated layer objects
layers = [
    LSTM(48, stateful=True),
    LSTM(12, stateful=True),
    Dense(1, activation=None, use_bias=False),
]

# instantiate a neural network model
model = NeuralNet(loss='mse', optimizer='adam', optimizer_args={'alpha': .001})

# fit the model to the training data
fitted_vals, loss = model.fit(
    X_train, y_train, epochs=1000, batch_size=None, verbose_rate=100, early_stopping_delta=1e-4)

# create validation forecast
val_forecast = model.get_forecast(val_len, init_input=X_test)

# forecast 12 steps beyond the historical data, using the last 
# sequence of X as the input
forecast = model.get_forecast(12, init_input=X[-1][np.newaxis, :, :])
```

If you’re familiar with Keras or TensorFlow, Sequential might feel simpler:

- In Keras, you often need to set `return_sequences` on recurrent layers. In Sequential, RNN and LSTM layers always return full sequences, and downstream Dense layers can process 3D inputs directly.

- Compressing from more input steps to fewer output steps (e.g. 60 -> 12) usually requires workarounds in Keras. Sequential handles this automatically with time-dimension compression. See `sequential/src/sequential/layers/dtc.py`.

For more information on layer initialization parameters and definitions, refer to each layer class in the `sequential/src/sequential/layers/` directory.

#### Transformer

Scripting is almost identical for transformers, except for model instantiation and one preprocessing parameter:

```python
import numpy as np
from sequential.models import Transformer
from sequential.preprocessing import *

'''
Load data, define variables (time_steps, val_len, etc,)
'''

# preprocessing is the same as the previous example, except with 
# autoregressive=True, targets are shifted by one so the model learns 
# to use time steps t1..tn to predict tn+1. The prediction window is 
# contained within the input window, enabled by masking during training.
X, X_train, X_test, y_train, y_val, scaler = preprocessing(
        X, val_len, time_steps, target_time_steps=target_time_steps, autoregressive=True)

model = Transformer(num_decoder_layers=3, d_model=20, units=64, num_heads=2, 
                    optimizer_args={'alpha': .001}, normalize=False, loss='mae', 
                    drop_rate=.1, apply_positional_encoding=False)

'''
model fitting and forecasting works the same as the previous example.
'''
```

#### train.py

If you’d like to use the same automated pipeline from `sequential/scripts/train.py` inside your own scripts, you can import its `run()` function. See the Command line section for more details on what `train.py` automates, and refer to the script itself for parameter options and definitions. 

(Requires `.[cli]` extras; see Installation section.)

## Tips

### Random seed for reproducibility 

Due to the random initialization of weights and biases in neural networks, there will be slight variation between fitted models with the same architecture. One option is to fit several models and average their forecasts. A simpler option, especially when testing different parameters, is to set a random seed with `np.random.seed(any_int_here)`. If you're using the command line script `sequential/scripts/train.py`, you can add a "random_seed" property with a corresponding integer to the JSON file, or pass that parameter directly when calling the `run()` function in your own scripts.

### Validation loss vs. training loss

It's normal for the validation loss to be higher than the training loss as the validation set is separated from the input data before training. However, too large of a difference is indicative of a problem. 

If the training loss is much lower than the validation loss, it suggests the model is overfitting to the training data. In the case of Dense, RNN, or LSTM models, try reducing layers or units, or adding a dropout layer. In the case of a Transformer, try reducing the number of decoder layers, reducing `d_model`, or increasing `drop_rate`.

If validation and training loss are roughly similar, the model may benefit from additional capacity (more layers, units, d_model, etc.).

### Loss plateaus during training

If the loss plateaus at a sub-optimal level for 100+ epochs, the optimizer may be stuck in a local minimum. Try:

- Using a different random seed (to vary initial weights and biases),
- Simplifying the model and training longer; or
- Increasing model size.

### Loss oscillation

If the loss oscillates from epoch to epoch instead of gradually decreasing, it may be that the learning rate is too high. Try passing a smaller alpha value in the `optimizer_args` of the model's initialization parameters.

## Testing

Sequential includes automated tests written with [pytest](https://docs.pytest.org/) to cover critical aspects of the framework, including gradient calculations, layer input/output shapes, and preprocessing transformations. 

To run the tests, first install the development dependencies:

```bash
pip install '.[dev]'
```

Then run:

```bash
pytest
```

from the project root to execute all tests in `sequential/tests/`.
