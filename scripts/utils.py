import numpy as np
import pandas as pd


def load_data(path):
    '''
    returns data as a pandas dataframe along with any 
    dates
    '''
    # read raw data into a dataframe
    df = pd.read_csv(path)
    # save dates if a date column is included in the input data
    dates = None
    date_freq = None
    if 'date' in df.columns.str.lower():
        date_name = 'date' if 'date' in df.columns else 'Date'
        dates = pd.to_datetime(df[date_name])
        date_freq = pd.infer_freq(dates)
        # drop dates from the inputs before processing
        df.drop(date_name, axis=1, inplace=True)

    return df, dates, date_freq


def scale_features(feat, scaler):
    '''
    Apply an existing MinMaxScaler to new feature values 
    (e.g., future values used during forecasting).

    Args
    ----
    feat: np.ndarray
        New feature values to scale, shaped (num_samples, num_features - 1).
    scaler: object 
        A previously fitted MinMaxScaler instance (fit on the training data).
    '''
    # start with a matrix of zeros with the same number
    # of columns as the input data to avoid a shape mismatch
    # error in the scaler
    feat_inv = np.zeros((len(feat), scaler.num_features))
    # place feature values in their corresponding locations
    # in the zeros matrix
    feat_inv[:, 1:] = feat
    # return the transformed features
    return scaler.transform(feat_inv)[:, 1:]


def inverse_target_scaling(y, scaler):
    ''' 
    Reverses scaling on the target variable.

    Args
    ----
    y: np.ndarray
        Array to perform the inverse transform on.
    scaler: object
        A previously fitted MinMaxScaler instance (fit on the training data).
    '''
    # start with a matrix of with the same number of columns
    # as the input data to avoid a shape mismatch error in the scaler
    y_inv = np.zeros((len(y), scaler.num_features))
    # place y in the first column
    y_inv[:, 0] = y.flatten()
    # invert scaling and return only the 1st column
    return scaler.inverse_transform(y_inv)[:, 0]
