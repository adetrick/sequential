import numpy as np
from sequential.preprocessing import MinMaxScaler


def preprocessing(X, val_len, time_steps, target_time_steps=None, autoregressive=False):
    '''
    Prepare inputs for model fitting and forecasting. 

    Scales input data to (0, 1), converts it into overlapping 
    3D sequences (num_sequences, time_steps, num_features), and 
    splits into training and validation sets.

    Args
    ----
    X: np.ndarray
        2D inputs shaped (num_samples, num_features).
    val_len: int
        Number of rows to reserve from the end of X for validation.
    time_steps: int
        Number of time steps per input sequence.
    target_time_steps: int
        Number of time steps per output sequence (y). Must be <= `time_steps`.
    autoregressive: bool
        If True (for Transformer models), targets are shifted by 1 
        so that time steps t1..tn predict tn+1 and `target_time_steps`
        are contained within that window. If False, targets are shifted 
        by the `time_steps` so that each full sequence predicts the next 
        window of `target_time_steps`.
    '''

    # scale the input to (0, 1)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # the target should be the first column
    y = X[:, 0]

    # turn inputs into a 3D array of sequences of length 'time_step',
    # with shape (num_sequences, time_steps, num_features)
    X = create_sequences(X, time_steps)

    # split into train and validation sets
    X_train, X_val, y_train, y_val = train_val_split(
        X, y, val_len, time_steps, target_time_steps, autoregressive=autoregressive)

    return X, X_train, X_val, y_train, y_val, scaler


def create_sequences(x, time_steps):
    '''
    Transforms inputs of shape (num_samples, num_features) into 
    a 3D array of shape (num_sequences, time_steps, num_features).
    Each output sequence is a sliding window of length `time_steps`

    Args
    ----
    x: np.ndarray
        2D array of shape (num_samples, num_features)
    time_steps: int
        Number of time steps in each output sequence
    '''
    x_len = x.shape[0] - time_steps + 1
    X = []
    for i in range(x_len):
        X.append(x[i:i+time_steps])
    return np.array(X)


def train_val_split(X, y, val_len, time_steps, target_time_steps=None, autoregressive=False):
    '''
    Splits 3D inputs into training and validation sets 

    Args
    ----
    X: np.ndarray
        3D inputs (num_sequences, time_steps, num_features).
    y: np.ndarray
        1D array of target values.
    val_len: int
        Number of rows to reserve from the end of X for validation.
    target_time_steps: int
        Number of time steps per output sequence (y). Must be <= `time_steps`.
    '''

    # ---- training set ----

    # create training set by slicing up to the validation length
    train = X[:-val_len]

    # determine the offset for y (target values). An offset of 1 is used
    # for autoregressive models (e.g. transformers) that apply a mask during
    # training so that all time steps from t1, t2...tn are used to predict
    # tn + 1. For non-autoregressive models, y is offset by the number of time
    # steps so that the each sequence is used to predict the next full sequence.
    # For example, the first 20 time steps are used to predict the following 20.
    offset = 1 if autoregressive else time_steps
    # apply the offset to X_train and y_train
    X_train = train[:-offset]
    if X_train.size == 0:
        raise ValueError(
            "The time_steps are too large for the given validation set length. Reduce one or both.")
    # keep only the 0th column in y_train in case the inputs
    # are multivariate
    y_train = train[offset:, :, 0][:, :, np.newaxis]

    # only keep the time steps we want to predict
    if target_time_steps is not None and target_time_steps < time_steps:
        if autoregressive:
            y_train = y_train[:, -target_time_steps:, :]
        else:
            y_train = y_train[:, :target_time_steps, :]

    # ---- validation set ----

    # use the last sequence of time steps in the training data as the
    # inputs for creating the validation forecast
    X_val = train[-1][np.newaxis, :, :]

    # use the last val_len values of the target y
    # for validation
    y_val = y[-val_len:]

    return X_train, X_val, y_train, y_val
