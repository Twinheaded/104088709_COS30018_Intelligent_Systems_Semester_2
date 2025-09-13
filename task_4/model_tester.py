import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.metrics import RootMeanSquaredError

from data_processor import load_and_process_data
from model_builder import create_dl_model

# This block ensures the code runs only when the script is executed directly
if __name__ == "__main__":
    
    # -- 1. Define Common Parameters --
    TICKER = "CBA.AX"
    START_DATE = "2015-01-01"
    END_DATE = "2023-12-31"
    N_STEPS = 50
    
    # -- 2. Load and Process Data (do this once) --
    data = load_and_process_data(TICKER, START_DATE, END_DATE, n_steps=N_STEPS)
    n_features = data["X_train"].shape[2]

    # -- 3. Define Experiments --
    experiments = [
        {"name": "LSTM_2_layers_128_units", "layer_type": "LSTM", "n_layers": 2, "layer_size": 128, "epochs": 20, "batch_size": 64},
        {"name": "LSTM_3_layers_128_units", "layer_type": "LSTM", "n_layers": 3, "layer_size": 128, "epochs": 20, "batch_size": 64},
        {"name": "GRU_2_layers_128_units", "layer_type": "GRU", "n_layers": 2, "layer_size": 128, "epochs": 20, "batch_size": 64},
        {"name": "LSTM_2_layers_256_units", "layer_type": "LSTM", "n_layers": 2, "layer_size": 256, "epochs": 20, "batch_size": 64},
        {"name": "LSTM_2_layers_128_units_50_epochs", "layer_type": "LSTM", "n_layers": 2, "layer_size": 128, "epochs": 50, "batch_size": 64},
        
        # Test a basic RNN
        {"name": "RNN_2_layers_128_units_20_epochs", "layer_type": "RNN", "n_layers": 2, "layer_size": 128, "epochs": 20, "batch_size": 64},
        
        # Test different batch sizes on the best-performing GRU model
        {"name": "GRU_2_layers_128_units_batch_32", "layer_type": "GRU", "n_layers": 2, "layer_size": 128, "epochs": 20, "batch_size": 32},
        {"name": "GRU_2_layers_128_units_batch_128", "layer_type": "GRU", "n_layers": 2, "layer_size": 128, "epochs": 20, "batch_size": 128},

        # Test different batch sizes on the best-performing LSTM model
        {"name": "LSTM_2_layers_128_units_50_epochs_batch_32", "layer_type": "LSTM", "n_layers": 2, "layer_size": 128, "epochs": 50, "batch_size": 32},
    ]

    # -- 4. Run Experiments --
    results = []
    
    for config in experiments:
        print(f"--- Running Experiment: {config['name']} ---")

        # Create the model using the config
        model = create_dl_model(
            n_layers=config["n_layers"], 
            layer_size=config["layer_size"], 
            layer_type=config["layer_type"], 
            input_shape=(N_STEPS, n_features)
        )
        
        # Train the model
        history = model.fit(
            data["X_train"], data["y_train"],
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            validation_data=(data["X_test"], data["y_test"]),
            verbose=0  # Set to 0 to keep the output clean
        )

        # Get the validation loss from the last epoch
        final_val_loss = history.history['val_loss'][-1]
        final_val_mae = history.history['val_mae'][-1]
        final_val_rmse = history.history['val_rmse'][-1]
        print(f"Final Validation Loss: {final_val_loss:.4f}")
        print(f"Final Validation MAE: {final_val_mae:.4f}")
        print(f"Final Validation RMSE: {final_val_rmse:.4f}")
        
        # Store results
        results.append({
            "name": config['name'],
            "validation_loss": final_val_loss,
            "validation_mae": final_val_mae,
            "validation_rmse": final_val_rmse
        })

    # -- 5. Display Results --
    print("\n--- Experiment Results ---")
    results_df = pd.DataFrame(results, columns=['name', 'validation_loss', 'validation_mae', 'validation_rmse'])
    results_df = results_df.sort_values(by="validation_loss")
    print(results_df)