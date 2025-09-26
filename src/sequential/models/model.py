import numpy as np
from sequential.layers import DenseTimeCompress
from sequential.metrics import mean_absolute_error, mean_squared_error


class Model:
    '''
    Base class for model training and evaluation. 

    This class should not be used directly. It is intended to 
    extend subclasses like NeuralNet or Transformer, which 
    must implement their own forward, backward, optimization, 
    and forecasting methods.
    '''

    def fit(
            self,
            X,
            y,
            epochs=100,
            batch_size=None,
            verbose_rate=10,
            early_stopping_delta=None,
            patience=10):
        '''
        Train the model on input data for a given number of epochs (iterations).

        Handles batches, loss calculation, backpropagation, parameter optimization,
        and optional early stopping. Returns the final predictions and
        full loss history.

        Args
        ----
        X: np.ndarray
            Training inputs with shape (num_sequences, time_steps, features).
        y: np.ndarray
            Target values with shape (num_sequences, time_steps, 1).
        epochs: int
            Number of training iterations.
        batch_size: int
            Number of samples per update step. If None, uses full batch.
        verbose_rate: int
            Frequency (in epochs) for printing the training loss. For example, a value 
            of 10 prints the loss every 10 epochs. Set to 0 to disable. Training 
            is usually slightly faster without verbosity.
        early_stopping_delta: float
            Minimum improvement in loss required to reset early stopping. 
            If provided, training stops when loss fails to improve by at 
            least this delta for `patience` consecutive epochs.
        patience: int
            Number of epochs without sufficient improvement before stopping.
            Only used if `early_stopping_delta` is set.
        '''
        if X.ndim != 3 or y.ndim != 3:
            raise ValueError('Inputs need to be 3D: (batch_size, time_steps, features)')
        self.X = X
        self.y = y
        self.dtc = DenseTimeCompress(y.shape[1]) if y.shape[1] < X.shape[1] else None
        # save the loss from each iteration
        loss_hist = []
        # track lowest loss
        best_loss = np.inf
        # track how many times the loss doesn't improve by a minimum
        # of the early stopping delta, if any
        early_stop_wait = 0
        # determine the number of batches
        num_batches = 1
        if batch_size is not None and batch_size < X.shape[0]:
            num_batches = int(np.floor(X.shape[0] / batch_size))
        # train the model
        for i in range(epochs):
            # reset running loss for each epoch
            running_loss = 0
            for b in range(num_batches):
                batch_start = b * batch_size if batch_size is not None else 0
                batch_end = batch_start + batch_size if batch_size is not None else None
                batch_X = self.process_input(X[batch_start:batch_end])
                batch_y = y[batch_start:batch_end]
                pred = self.predict(batch_X, False)
                # add to the running loss for the current epoch
                running_loss += self.cost(batch_y, pred)
                # calculate gradients
                self.back_propagation(pred, batch_y)
                # update parameters
                self.optimize(i)
            # take the average loss across batches
            loss = running_loss / num_batches
            # add the current loss to the history
            loss_hist.append(loss)
            # print the loss, if applicable
            if verbose_rate > 0 and (i % verbose_rate == 0 or i == epochs - 1):
                print(f"Epoch {i}, loss: {loss}")
            # if there's an early stopping delta, determine whether an improvement
            # has been made in the loss
            if early_stopping_delta:
                if loss < (best_loss - early_stopping_delta):
                    best_loss = loss
                    early_stop_wait = 0
                else:
                    # no improvement
                    early_stop_wait += 1
                # break if the patience threshold is reached
                if early_stop_wait == patience:
                    print(f"Early stopping at epoch {i}, loss {loss}")
                    break

        return pred, loss_hist

    def get_forecast(self, steps, init_input=None, features=None):
        '''
        Args
        ----
        steps: int
            number of time steps to predict into the future
        init_input: np.ndarray 
            initial input for starting the forecast, shaped 
            (1, time_steps, features)
        features: np.ndarray
            future feature values to append to the prediction at 
            each forecast step with shape (time_steps, features). 
            The features dimension needs to match X's num features
        '''
        self.validate_forecast_features(features)
        # store forecast values in a list
        forecast = []
        # use the last batch for the initial input as a fallback
        inputs = init_input if init_input is not None else self.X[-1][np.newaxis, :, :]
        # iterate through the steps, creating one prediction at a time
        for i in range(steps):
            inputs_i = self.process_input(inputs)
            pred = self.predict(inputs_i, True)
            # keep the first predicted value
            pred = pred[:, :1, :]
            # save the prediction
            forecast.append(pred.flatten())
            if i < steps - 1:
                if features is not None:
                    feat_i = features[i, :].reshape(1, 1, -1)
                    # concat features to the prediction along the
                    # last axis
                    pred = np.concatenate([pred, feat_i], axis=-1)
                # append the prediction to the inputs for
                # the next forecast step
                inputs = np.concatenate([inputs[:, 1:, :], pred], axis=1)

        return np.array(forecast)

    def process_input(self, X):
        if hasattr(self, 'embed_layer') and self.embed_layer is not None:
            X = self.embed_layer(X)
        if hasattr(self, 'apply_positional_encoding') and self.apply_positional_encoding:
            X += self.positional_encoding(X.shape[1], X.shape[2])
        return X

    def cost(self, y, pred):
        '''
        Computes the loss between y (targets) and pred (predictions)

        Supports 'mse' (mean squared error) and 'mae' (mean absolute error)
        '''
        if self.loss == 'mse':
            return mean_squared_error(y, pred)
        elif self.loss == 'mae':
            return mean_absolute_error(y, pred)

    def dcost(self, y, pred):
        '''
        Computes the gradient of the loss function with respect to predictions

        Supports 'mse' (mean squared error) and 'mae' (mean absolute error)
        '''
        if self.loss == 'mse':
            return -2 * (y - pred) / len(y)
        elif self.loss == 'mae':
            return -1 * np.sign(y - pred) / len(y)

    def positional_encoding(self, seq_len, d_model, n=10000):
        '''
        Generate sinusoidal positional encodings for a sequence.

        Adds position information to embeddings so the model can
        distinguish order. Each position is represented by sine 
        and cosine functions at different frequencies.

        Args
        ----
        seq_len: int
            Length of the input sequence.
        d_model: int
            Dimensionality of the model/embeddings. Must be even 
            (half sine, half cosine).
        n: int, optional
            Wavelength scaling factor (default: 10000).
        '''
        # row indices
        ri = np.arange(seq_len)[:, np.newaxis]
        # column indices, divided by two to create pairs
        # of the same index number since each one is
        # assigned a sin and cosine value
        ci = np.arange(d_model)[np.newaxis, :] // 2
        # calc the encodings
        pe = ri / (n ** ((2*(ci))/d_model))
        # apply sin to even indices
        pe[:, ::2] = np.sin(pe[:, ::2])
        # apply cos to odd indices
        pe[:, 1::2] = np.cos(pe[:, 1::2])

        return pe[np.newaxis, :, :]

    def validate_forecast_features(self, features):
        # check if the input contains features
        num_features = self.X.shape[-1] - 1
        # validation if features are present
        if num_features > 0:
            if features is None:
                raise ValueError(
                    f"features must be provided as the model was trained with {num_features} features")
            # forecast feature columns must match the training data
            assert num_features == features.shape[1]
