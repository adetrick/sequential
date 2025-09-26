import numpy as np


class MinMaxScaler():
    '''
    Scales input data to the range (0, 1).

    Formula:
        X_scaled = (X - X_min) / (X_max - X_min)

    Notes
    -----
    - The scaler computes per-feature minimum and maximum values 
      during `fit`.
    - If a feature has zero range (max == min), its range is set 
      to 1 to avoid division by zero.
    '''

    def __init__(self):
        self._fit = False

    def fit(self, x):
        '''
        Compute per-feature minimum, maximum, and range.

        Args
        ----
        x: np.ndarray
            2D array of shape (num_samples, num_features).
        '''
        if type(x) != np.ndarray:
            raise ValueError("Input data must be a numpy array")
        self.num_features = x.shape[-1]
        self.max_val = np.max(x, axis=0)
        self.min_val = np.min(x, axis=0)
        self.range = self.max_val - self.min_val
        # set zero ranges to 1 to avoid division by zero
        if type(self.range) == np.ndarray:
            self.range[self.range == 0] = 1
        elif self.range == 0:
            self.range = 1
        self._fit = True

        return self

    def fit_transform(self, x):
        '''
        Fit to input data, then transform it.

        Args
        ----
        x: np.ndarray
            2D array of shape (num_samples, num_features).
        '''
        return self.fit(x).transform(x)

    def transform(self, x):
        '''
        Scale input data to the range (0, 1).

        Args
        ----
        x: np.ndarray
            2D array of shape (num_samples, num_features).
        '''
        self.validate_transform(x)
        return (x - self.min_val) / self.range

    def inverse_transform(self, x):
        '''
        Undo scaling and return original values.

        Args
        ----
        x: np.ndarray
            2D array of shape (num_samples, num_features).
        '''
        self.validate_transform(x)
        return (x * self.range) + self.min_val

    def validate_transform(self, x):
        '''
        Validates the scaler has been fit and the input shape matches.
        '''
        if not self._fit:
            raise ValueError(
                "This MinMaxScaler has not been fit on input data. Call fit() or fit_transform()")
        if self.num_features != x.shape[-1]:
            raise ValueError(
                f"Input feature dimension ({x.shape[-1]}) does not match fitted dimension ({self.num_features}).")
