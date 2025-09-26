import pytest
import numpy as np
from sequential.layers import *
from sequential.layers.transformer import *


def calc_numerical_gradient(f, x, eps=1e-5):
    '''
    Applies central finite difference to calculate 
    the numerical gradient of x.

    Args
    ----
    f: function
        Function that returns scalar loss given input x.
    x: np.ndarray
        Array of parameters.
    '''
    grad = np.zeros_like(x)
    grad_rav = grad.ravel()
    # flatten x in order to iterate over each value.
    # np.ravel() usually returns a view, so modifications here
    # directly change x.
    x_rav = x.ravel()

    for i in range(x_rav.size):
        orig_val = x_rav[i]
        x_rav[i] = orig_val + eps
        plus = f(x)
        x_rav[i] = orig_val - eps
        minus = f(x)
        grad_rav[i] = (plus - minus) / (2 * eps)
        # restore orig value
        x_rav[i] = orig_val

    return grad


def numerical_gradient_check(layer, X, eps=1e-5):
    '''
    Compares analytical vs numerical gradients for the weights, bias, 
    and inputs of a layer object (Dense, RNN, LSTM, DenseTimeCompress,
    LayerNorm, TransformerDense, and Attention).
    '''

    grad_res = {'inputs': {}}

    # ---- Analytical gradients ----

    # Forward pass to build weights/bias matrices and
    # populate the cache needed for backpropagation
    outputs = layer(X)

    # update grad res with the trainable params from this layer
    for k, vals in layer.trainable_params.items():
        if vals is None:
            continue
        grad_res[k] = {}

    # Backward pass to compute analytical gradients.
    # Ones for upstream gradients so that the output gradients
    # only reflect calculations from within the backward pass
    upstream_grad = np.ones_like(outputs)
    grad_res['inputs']['analytical_grad'] = layer.backward(upstream_grad)

    # analytical gradient for weights/bias
    for k, grads in layer.trainable_params_grad.items():
        grad_res[k]['analytical_grad'] = grads

    # ---- Numerical gradients ----

    # numerical gradient w.r.t. inputs. X_in is adjusted directly
    # during differencing, so it's passed to the layer each time.
    def f(X_in): return np.sum(layer(X_in))
    grad_res['inputs']['numerical_grad'] = calc_numerical_gradient(f, X, eps)

    # numerical gradient for weights/bias
    for k, vals in layer.trainable_params.items():
        if vals is None:
            continue
        # X stays fixed while each parameter is adjusted during differencing.
        def f(param): return np.sum(layer(X))
        grad_res[k]['numerical_grad'] = calc_numerical_gradient(f, vals, eps)

    return grad_res


def test_layer_gradients():
    '''
    Verifies the correctness of analytically computed gradients for layers 
    with trainable parameters.

    For each layer, the analytically derived gradients are compared to 
    numerically approximated gradients using central finite differences. 
    This ensures that the backward() implementations are correctly 
    computing the gradients needed for training.

    Layers tested include Dense, RNN, LSTM, DenseTimeCompress, LayerNorm, 
    TransformerDense, and Attention.
    '''
    np.random.seed(0)
    # use a minimal input (1 batch, 1 time step, 1 feature)
    # to test gradient correctness
    X = np.random.random((1, 1, 1))
    num_units = 1

    # test all layers with trainable weights/biases
    layers = [
        Dense(units=num_units, use_bias=True),
        RNN(units=num_units, use_bias=True),
        LSTM(units=num_units, use_bias=True),
        DenseTimeCompress(num_units),
        LayerNorm(),
        TransformerDense(num_units),
        Attention(num_units),
    ]

    for layer in layers:
        res = numerical_gradient_check(layer, X)

        for name, grads in res.items():
            analytical_grad = grads['analytical_grad']
            numerical_grad = grads['numerical_grad']

            assert np.allclose(analytical_grad, numerical_grad, atol=1e-6, rtol=1e-4), \
                f"Gradient check failed for {name} in {type(layer).__name__} layer: max diff={np.max(np.abs(analytical_grad - numerical_grad))}"


def test_layer_shapes_and_grad_flow():
    '''
    Sanity test for layer forward/backward consistency. 

    Verifies that:
      - The output shape of each layer's forward/backward pass 
        matches the expected shape (typically the input shape).
      - No NaN or Inf values appear in either forward or 
        backward passes.

    This helps catch errors in layer shape handling or 
    gradient propagation.
    '''

    # initialize X with (batches, time_steps, features).
    # Multiple values in each dimension test shape handling
    # and gradient flow more robustly.
    num_units = 10
    X = np.random.random((5, 15, num_units))

    layers = [
        Dense(units=num_units, use_bias=True),
        RNN(units=num_units, use_bias=True),
        LSTM(units=num_units, use_bias=True),
        # for DenseTimeCompress: output shape matches input,
        # except the time dimension is reduced to `units`.
        DenseTimeCompress(num_units),
        LayerNorm(),
        TransformerDense(num_units),
        DecoderLayer(2, num_units, .1, normalize=True),
        Attention(num_units),
        Dropout(),
    ]

    for layer in layers:
        fail_message = f"Assert check for {type(layer).__name__} layer failed"

        # forward pass to create the output
        output = layer(X)

        if isinstance(layer, DenseTimeCompress):
            # for DenseTimeCompress, forward shape should equal input
            # shape except with the time dimensions compressed to the
            # number of units
            assert output.shape == (X.shape[0], layer.units, X.shape[2])
        else:
            # forward output shape should equal input shape
            assert output.shape == X.shape, fail_message

        # backward pass should return same shape as input
        grad_upstream = np.random.randn(*output.shape)
        grad_inputs = layer.backward(grad_upstream)
        assert grad_inputs.shape == X.shape, fail_message

        # should be no nans/infs
        assert np.all(np.isfinite(output)), fail_message
        assert np.all(np.isfinite(grad_inputs)), fail_message
