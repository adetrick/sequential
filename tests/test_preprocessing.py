import pytest
import numpy as np
from sequential.preprocessing import *


def test_preprocessing():
    '''
    Tests preprocessing output.

    Verifies that:
        - the outputs are 3D
        - the training and target time steps line up
        - the training and and validation sets are split correctly
        - the test set for validation forecasting equals the 
          last sequence of the training data 
        - the offset has been applied correctly to the target

    '''
    time_steps = 25
    target_time_steps = 10
    val_len = 10
    X = np.random.randn(100, 2)
    X_shape = X.shape

    X, X_train, X_test, y_train, y_val, scaler = preprocessing(
        X, val_len, time_steps, target_time_steps=target_time_steps, autoregressive=False)

    res_dict = {
        'X': X,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
    }

    for name, arr in res_dict.items():
        # X and y should be returned with 3 dimensions (num_sequences, time_steps, num_features),y_val doesn't need to be 3D
        if name != 'y_val':
            assert arr.ndim == 3, f"{name} must be 3D, got {arr.ndim}D"

        if 'X_' in name:
            # the output time steps for the training inputs (X) should equal `time_steps`
            assert arr.shape[1] == time_steps, f"{name} must have {time_steps} time steps in the second dimension, got {arr.shape[1]}"

            # make sure the number of features has been preserved in the training
            # inputs X
            assert arr.shape[-1] == X_shape[1], f"{name} must have {X_shape[1]} features in the last dimension, got {arr.shape[-1]}"

        elif name == 'y_train':
            # the target should only have 1 feature in the last dimension
            assert arr.shape[-1] == 1

            # the output time steps for the target (y) should equal `target_time_steps`
            assert arr.shape[1] == target_time_steps, f"{name} must have {target_time_steps} time steps in the second dimension, got {arr.shape[1]}"

    # training set lengths should match
    assert y_train.shape[0] == X_train.shape[0]

    # validation set length should equal `val_len`
    assert y_val.size == val_len

    # y_val should equal the last sequence of values in X equal to `val_len`
    assert (y_val == X[-val_len:, -1, 0]).all()

    # X_test should equal the last sequence of the training portion of X
    assert (X_test == X[:-val_len][-1][np.newaxis, :, :]).all()

    # y_train should equal X sliced by `time_steps` up to the start of
    # the validation set
    assert (y_train == X[time_steps:-val_len, :target_time_steps, :1]).all()

    # X_train should equal X up to the length of the validation
    # set and `time_steps`
    assert (X_train == X[:-val_len - time_steps, :, :]).all()

    # target offset: for each sequence in X_train, y_train should consist
    # of values from the next sequence (up to `target_time_steps`)
    assert (X_train[time_steps:, :target_time_steps, :1] == y_train[:-time_steps, :, :]).all()


def test_autoregressive_preprocessing():
    '''
    Tests preprocessing output for autoregressive tasks.

    The only difference in output compared to when 
    autoregressive is False is the target offset, which 
    is what we'll test.

    '''

    time_steps = 25
    target_time_steps = 10
    val_len = 10
    X = np.random.randn(100, 2)

    X, X_train, X_test, y_train, y_val, scaler = preprocessing(
        X, val_len, time_steps, target_time_steps=target_time_steps, autoregressive=True)

    # When `autoregressive` is True, the target offset should equal 1. When
    # `target_time_steps` < `time_steps`, the target values are sliced
    # from the end of each training sequence
    assert (X_train[1:, -target_time_steps:, :1] == y_train[:-1, :, :]).all()


def test_scaler():
    '''
    Verifies that the MinMaxScaler transforms 
    the input array to the range of (0, 1)
    '''

    X = np.random.randn(50) + 20
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    assert ((X_scaled >= 0) & (X_scaled <= 1)).all()
