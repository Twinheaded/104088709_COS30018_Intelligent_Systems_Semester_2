import numpy as np
import matplotlib.pyplot as plt

# Import our custom functions from the other files
from data_processor import load_and_process_data
from model_builder import create_model
from model_builder import create_dl_model


# This block ensures the code runs only when the script is executed directly
if __name__ == "__main__":
    
    # -- 1. Define Parameters --
    TICKER = "CBA.AX"
    START_DATE = "2015-01-01"
    END_DATE = "2023-12-31"
    N_STEPS = 50  # Use 50 days of historical data to predict the next day
    
    # -- 2. Load and Process Data --
    # Call the function from data_processor.py
    data = load_and_process_data(TICKER, START_DATE, END_DATE, n_steps=N_STEPS)

    # -- 3. Create and Train the Model --
    
    n_features = data["X_train"].shape[2]
    
    # Example: Create an LSTM model
    model = create_dl_model(n_layers=2, layer_size=128, layer_type='LSTM', input_shape=(data["X_train"].shape[1], n_features))

    # # Example: Create a GRU model
    # model = create_dl_model(n_layers=3, layer_size=256, layer_type='GRU', input_shape=(50, 1))
    
    # # Call the function from model_builder.py
    # model = create_model(N_STEPS, data["X_train"].shape[2])
    

    
    # Train the model
    model.fit(data["X_train"], data["y_train"],
              batch_size=64,
              epochs=20,
              validation_data=(data["X_test"], data["y_test"]))

    # -- 4. Make and Inverse-Transform Predictions --
    predictions = model.predict(data["X_test"])
    
    # Use the saved 'Close' scaler to transform predictions back to real prices
    close_scaler = data['scalers']['Close']
    y_test_real = close_scaler.inverse_transform(data['y_test'].reshape(-1, 1))
    predictions_real = close_scaler.inverse_transform(predictions)

    # -- 5. Plot the Results --
    original_df = data['df']
    train_data = original_df.iloc[:len(data['X_train'])]
    valid_data = original_df.iloc[len(data['X_train']):len(data['X_train']) + len(y_test_real)]
    valid_data['Predictions'] = predictions_real

    plt.figure(figsize=(16,8))
    plt.title(f'Model for {TICKER}')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(train_data['Close'])
    plt.plot(valid_data[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()