import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

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