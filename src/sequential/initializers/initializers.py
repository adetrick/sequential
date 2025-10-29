import numpy as np


def glorot_uniform(fan_in, fan_out):
    '''
    Args
    ----
    fan_in: int
        Number of input paths towards the neuron (number of 
        input units)
    fan_out: int
        Number of output paths towards the neuron (number of 
        output units)
    '''
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(low=-limit, high=limit, size=(fan_in, fan_out))


def glorot_normal(fan_in, fan_out):
    '''
    Args
    ----
    fan_in: int
        Number of input paths towards the neuron (number of 
        input units)
    fan_out: int
        Number of output paths towards the neuron (number of 
        output units)
    '''
    std = np.sqrt(2 / (fan_in + fan_out))
    return np.random.normal(loc=0, scale=std, size=(fan_in, fan_out))


def he_uniform(fan_in, fan_out):
    '''
    Args
    ----
    fan_in: int
        Number of input paths towards the neuron (number of 
        input units)
    fan_out: int
        Number of output paths towards the neuron (number of 
        output units)
    '''
    limit = np.sqrt(6 / fan_in)
    return np.random.uniform(low=-limit, high=limit, size=(fan_in, fan_out))


def he_normal(fan_in, fan_out):
    '''
    Args
    ----
    fan_in: int
        Number of input paths towards the neuron (number of 
        input units)
    fan_out: int
        Number of output paths towards the neuron (number of 
        output units)
    '''
    std = np.sqrt(2 / fan_in)
    return np.random.normal(loc=0, scale=std, size=(fan_in, fan_out))
