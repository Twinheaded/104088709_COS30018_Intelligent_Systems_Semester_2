import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import deque

def load_and_process_data(
    ticker,
    start_date,
    end_date,
    n_steps=50,
    feature_columns=['Open', 'High', 'Low', 'Close', 'Volume'],
    test_size=0.2,
    split_by_date=True,
    shuffle=True,
    lookup_step=1
):

    # Define the folder name for our data cache
    cache_folder = "data_cache"

    # Create the folder if it doesn't already exist
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    # Create a unique filename and join it with the folder path
    filename = os.path.join(cache_folder, f"{ticker}_{start_date}_{end_date}.csv")

    # Check if data is cached locally
    if os.path.exists(filename):
        print(f"Loading data from local file: {filename}")
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
    else:
        # Download data if not cached
        print(f"Downloading data for {ticker}...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if isinstance(df.columns, pd.MultiIndex):
            # If columns are multi-level (e.g., [('Close', 'CBA.AX')]), flatten them.
            df.columns = df.columns.get_level_values(0)

        df.to_csv(filename, index_label='Date')

    # Prepare result dictionary and handle missing values
    result = {'df': df.copy()}
    df.ffill(inplace=True)

    # Scale features and store scalers
    scalers = {}
    for col in feature_columns:
        scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler
    result['scalers'] = scalers

    # Create the target 'future' column
    df['future'] = df['Close'].shift(-lookup_step)
    df.dropna(inplace=True)

    # Create sequences of data for the LSTM model
    sequences = deque(maxlen=n_steps)
    sequence_data = []
    for entry, target in zip(df[feature_columns].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # Convert to numpy arrays
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    X = np.array(X)
    y = np.array(y)

    # Split data into training and testing sets
    if split_by_date:
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"]  = X[train_samples:]
        result["y_test"]  = y[train_samples:]
    else:
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(
            X, y, test_size=test_size, shuffle=shuffle
        )

    return result