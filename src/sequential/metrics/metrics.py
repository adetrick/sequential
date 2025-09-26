import numpy as np


def mean_absolute_error(x, y):
    return np.mean(abs(x.flatten() - y.flatten()))


def mean_squared_error(x, y):
    return np.mean((x.flatten() - y.flatten()) ** 2)
