import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout
from tensorflow.keras.metrics import RootMeanSquaredError

def create_model(sequence_length, num_features):
    """
    Creates, compiles, and returns the LSTM model.
    """
    model = Sequential()
    # First LSTM layer with return_sequences=True to pass sequences to the next layer
    model.add(LSTM(128, return_sequences=True, input_shape=(sequence_length, num_features)))
    # Second LSTM layer
    model.add(LSTM(64, return_sequences=False))
    # A fully connected dense layer
    model.add(Dense(25))
    # The output layer with a single unit for the prediction
    model.add(Dense(1))

    # Compile the model for training
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def create_dl_model(n_layers: int, layer_size: int, layer_type: str, input_shape: tuple, dropout_rate: float = 0.2):
    """
    Dynamically creates a Deep Learning model.

    Args:
        n_layers (int): The number of hidden layers in the model.
        layer_size (int): The number of units in each hidden layer.
        layer_type (str): The type of recurrent layer to use ('LSTM', 'GRU', or 'RNN').
        input_shape (tuple): The shape of the input data (e.g., (sequence_length, n_features)).
        dropout_rate (float): The dropout rate for regularization.

    Returns:
        A compiled TensorFlow Keras model.
    """
    model = Sequential()
    
    # Add the recurrent layers
    for i in range(n_layers):
        # The 'return_sequences' parameter needs to be True for all but the last recurrent layer
        # This is because subsequent recurrent layers need a sequence as input, not a single vector.
        return_sequences = (i < n_layers - 1)
        
        if i == 0:
            # The first layer needs the input_shape parameter
            if layer_type.upper() == 'LSTM':
                model.add(LSTM(layer_size, return_sequences=return_sequences, input_shape=input_shape))
            elif layer_type.upper() == 'GRU':
                model.add(GRU(layer_size, return_sequences=return_sequences, input_shape=input_shape))
            elif layer_type.upper() == 'RNN':
                model.add(SimpleRNN(layer_size, return_sequences=return_sequences, input_shape=input_shape))
        else:
            # Subsequent layers do not need the input_shape parameter
            if layer_type.upper() == 'LSTM':
                model.add(LSTM(layer_size, return_sequences=return_sequences))
            elif layer_type.upper() == 'GRU':
                model.add(GRU(layer_size, return_sequences=return_sequences))
            elif layer_type.upper() == 'RNN':
                model.add(SimpleRNN(layer_size, return_sequences=return_sequences))

        # Add a dropout layer for regularization to prevent overfitting.
        # This randomly sets a fraction of input units to 0 at each update during training.
        model.add(Dropout(dropout_rate))

    # Add the final output layer. A Dense layer is used for this.
    # The number of units is 1, as we are predicting a single value (e.g., stock price).
    model.add(Dense(1))

    # Compile the model. The optimizer and loss function can be changed based on the experiments.
    model.compile(optimizer='adam', 
                  loss='mean_squared_error', 
                  metrics=['mae', RootMeanSquaredError(name='rmse')])
    
    return model