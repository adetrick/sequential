import numpy as np
import keras


def build_keras_decoder_only_transformer(input_shape,
                                         target_time_steps=None,
                                         d_model=50,
                                         num_heads=1,
                                         num_layers=6,
                                         ff_units=64,
                                         loss='mse',
                                         learning_rate=.001,
                                         normalize=False
                                         ):
    '''

    Build a decoder-only Transformer model with Keras for sequential data.

    The model projects raw inputs into a d_model-dimensional space,
    passes them through stacked Transformer decoder layers with
    causal masking, and projects the outputs down to a single value
    per time step.

    Args
    ----
    input_shape: tuple of int
        Shape of the input excluding the batch dimension,
        given as (time_steps, num_features).
    target_time_steps : int or None
        Desired number of output time steps. If None, defaults
        to the input sequence length.
    d_model: int
        Dimensionality of the model's vector representations
        (i.e. embeddings) of the raw inputs.
    num_heads: int
        Number of heads in each self-attention block.
    num_layers: int
        Number of stacked decoder layers in the Transformer.
    ff_units: int
        Number of hidden units in the feed-forward sublayer
        within each decoder layer.
    loss: str
        Loss function to minimize. Options: 'mse', 'mae'.
    learning_rate: float
        Learning rate for parameter updates using the Adam optimizer.
    '''
    inputs = keras.layers.Input(shape=input_shape)

    # embedding / projection
    x = keras.layers.Dense(d_model, use_bias=True)(inputs)

    # stack decoder layers
    for _ in range(num_layers):

        # --- Self-attention ---

        attn_out = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=0.0,
        )(x, x, use_causal_mask=True)
        # residual connection
        x = keras.layers.Add()([x, attn_out])
        if normalize:
            x = keras.layers.LayerNormalization()(x)

        # --- Feed-forward (two linear layers with ReLU in between) --

        # first projection layer plus ReLU
        ff = keras.layers.Dense(ff_units, activation="relu")(x)
        # second projection back to model dim
        ff = keras.layers.Dense(d_model, activation=None)(ff)
        # residual connection
        x = keras.layers.Add()([x, ff])
        if normalize:
            x = keras.layers.LayerNormalization()(x)

    # output projection
    x = keras.layers.Dense(1, use_bias=False, activation=None)(x)

    input_time_steps = input_shape[0]
    # if there are fewer time steps in the target than in the input
    # sequence, project the output down to the number of
    # target time steps
    if target_time_steps and target_time_steps < input_time_steps:
        x = keras.layers.Reshape((input_time_steps,))(x)
        x = keras.layers.Dense(target_time_steps, activation=None)(x)
        x = keras.layers.Reshape((target_time_steps, 1))(x)

    model = keras.Model(inputs, x)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=loss,
    )
    return model


def generate_keras_forecast(steps, X, model):
    '''
    Generate an autoregressive forecast using a fitted Keras model.

    At each step, the model predicts the full output sequence for the
    current input window. The first predicted time step is kept,
    appended to the forecast history, and fed back into the input
    sequence for the next iteration.

    Args
    ----
    steps : int
        Number of future time steps to forecast.
    X : np.ndarray
        Initial input sequence of shape (1, time_steps, features).
    model : keras.Model
        A trained Keras model created by build_decoder_only_transformer().

    '''
    forecast = []

    for i in range(steps):
        pred = model.predict(X, verbose=0)
        # keep the first predicted value
        pred = pred[:, :1, :]
        # save the prediction
        forecast.append(pred.flatten())
        if i < steps - 1:
            # append the prediction to the inputs for
            # the next forecast step
            X = np.concatenate([X[:, 1:, :], pred], axis=1)

    return np.array(forecast)
