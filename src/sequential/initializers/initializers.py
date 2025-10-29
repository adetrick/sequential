import numpy as np


def glorot_uniform(fan_in, fan_out):
    '''
    Glorot (Xavier) uniform initializer.

    Weights are sampled from a uniform distribution in the range:
        [-limit, limit], where limit = sqrt(6 / (fan_in + fan_out))

    This keeps the variance of activations more stable across layers 
    and reduces vanishing or exploding gradients, especially in 
    networks with Sigmoid or Tanh activations.      

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
    Glorot (Xavier) normal initializer.

    Weights are sampled from a normal distribution with:
        mean = 0
        std  = sqrt(2 / (fan_in + fan_out))

    Helps stabilize gradient flow in networks with Sigmoid or 
    Tanh activations.

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
    He uniform initializer.

    Weights are sampled from a uniform distribution in the range:
        [-limit, limit], where limit = sqrt(6 / fan_in)

    Designed for ReLU and related activations, where preserving
    forward variance requires scaling by fan_in only.

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
    He normal initializer.

    Weights are sampled from a normal distribution with:
        mean = 0
        std  = sqrt(2 / fan_in)

    Best suited for layers with ReLU or variants, helping
    prevent vanishing/exploding gradients.

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
