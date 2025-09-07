import pandas as pd

# Import our custom functions
from data_processor import load_and_process_data
from visualizer import plot_candlestick_chart, plot_price_distribution_boxplot

# This block ensures the code runs only when the script is executed directly
if __name__ == "__main__":

    # -- 1. Define Parameters for Analysis --
    TICKER = "CBA.AX"
    START_DATE = "2022-01-01"
    END_DATE = "2024-12-31"

    # -- 2. Load the Data --
    # We only need the original DataFrame, so we can ignore the processed data for now.
    data = load_and_process_data(TICKER, START_DATE, END_DATE, n_steps=1)
    original_df = data['df']

    # -- 3. Display Visualizations --
    print(f"Displaying charts for {TICKER}...")

    # Change the n_days for the desired days candlestick chart
    plot_candlestick_chart(original_df, TICKER, n_days=14)
    
    plot_price_distribution_boxplot(original_df, TICKER, n_days=21)

    print("Analysis complete.")