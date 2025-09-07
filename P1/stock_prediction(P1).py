# stock_prediction.py - Add this function at the top

import numpy as np
import pandas as pd
from collections import deque
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

def get_stock_data_robust(ticker):
    """Robust data fetching with fallbacks"""
    print(f"Attempting to fetch data for ticker: {ticker}")
    
    # Try yfinance first (most reliable)
    try:
        import yfinance as yf
        print("Trying yfinance...")
        
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y")
        
        if df.empty:
            raise ValueError("Empty dataframe returned")
            
        print(f"Raw yfinance columns: {list(df.columns)}")
        
        # Standardize column names to lowercase and remove spaces
        df.columns = df.columns.str.lower().str.replace(' ', '')
        
        # Create adjclose from close if it doesn't exist
        if 'adjclose' not in df.columns and 'close' in df.columns:
            df['adjclose'] = df['close']
            print("Created 'adjclose' from 'close' column")
        
        if 'adjclose' not in df.columns:
            raise ValueError("Could not find or create 'adjclose' column")
        
        print(f"✅ yfinance success: {len(df)} rows of data")
        print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"Final columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"❌ yfinance failed: {e}")
    
    # Try yahoo_fin as fallback
    try:
        print("Trying yahoo_fin fallback...")
        import yahoo_fin.stock_info as si
        from datetime import datetime, timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        df = si.get_data(ticker, 
                        start_date=start_date.strftime('%m/%d/%Y'), 
                        end_date=end_date.strftime('%m/%d/%Y'))
        
        print(f"✅ yahoo_fin success: {len(df)} rows of data")
        return df
        
    except Exception as e:
        print(f"❌ yahoo_fin failed: {e}")
    
    raise Exception(f"All data sources failed for ticker: {ticker}")


def shuffle_in_unison(a, b):
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


def load_data(ticker, n_steps, scale=True, split_by_date=True, shuffle=True, lookup_step=1, test_size=0.2, feature_columns=['adjclose']):
    """
    Loads data from Yahoo Finance
    """
    # Use the robust data fetching function
    df = get_stock_data_robust(ticker)
    
    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['df'] = df.copy()
    
    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
    
    # add date as a column (yfinance uses index for dates)
    if "date" not in df.columns:
        df["date"] = df.index
        
    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler

    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['adjclose'].shift(-lookup_step)

    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    # drop NaNs
    df.dropna(inplace=True)

    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices that are not available in the dataset
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    # add to result
    result['last_sequence'] = last_sequence
    
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    if split_by_date:
        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"]  = X[train_samples:]
        result["y_test"]  = y[train_samples:]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:
        # split the dataset randomly
        from sklearn.model_selection import train_test_split
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, 
                                                                                test_size=test_size, shuffle=shuffle)

    # get the list of test set dates
    dates = result["X_test"][:, -1, -1]
    # retrieve test features from the original dataframe
    # Fix: Handle the case where date might be in index or column
    if "date" in result["df"].columns:
        result["test_df"] = result["df"].loc[result["df"]["date"].isin(dates)]
    else:
        # If date is in index, convert dates to datetime and match with index
        try:
            dates_as_datetime = pd.to_datetime(dates)
            result["test_df"] = result["df"].loc[result["df"].index.isin(dates_as_datetime)]
        except:
            # Fallback: use the last portion of data for testing
            test_size_samples = len(result["X_test"])
            result["test_df"] = result["df"].tail(test_size_samples)
    
    # remove duplicated dates in the testing dataframe
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # remove dates from the training/testing sets & convert to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

    return result


def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    model = Sequential()
    
    for i in range(n_layers):
        if i == 0:
            # first layer - use input_shape instead of batch_input_shape
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True, input_shape=(sequence_length, n_features))))
            else:
                model.add(cell(units, return_sequences=True, input_shape=(sequence_length, n_features)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    
    return model