import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

def plot_candlestick_chart(df, ticker, n_days=1):
    """
    Displays a candlestick chart for the given stock data.

    This function can resample the data to show each candle as an aggregate
    of 'n' trading days.

    Args:
        df (pd.DataFrame): The input DataFrame containing the stock data.
            It must have 'Open', 'High', 'Low', and 'Close' columns.
        ticker (str): The stock ticker symbol (e.g., "AAPL") to be used in the chart title.
        n_days (int): The number of trading days to aggregate into a single candlestick.
            Default is 1, which means one candle per day.
    """

    # --- Requirement: Resample data for n trading days ---
    # The task requires an option to have each candlestick represent 'n' trading days.
    # We use the pandas resample() method for this.

    # First, we define the aggregation logic. For each period of 'n' days, we need:
    # - The 'Open' of the first day
    # - The 'High' of the entire period
    # - The 'Low' of the entire period
    # - The 'Close' of the last day
    ohlc_aggregation = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }

    # Resample the DataFrame. The string f'{n_days}D' creates a frequency string,
    # e.g., '5D' for 5 days. We then apply our custom aggregation logic.
    resampled_df = df.resample(f'{n_days}D').agg(ohlc_aggregation)

    # Drop any rows that might be empty after resampling (e.g., weekends with no data)
    resampled_df.dropna(inplace=True)

    # --- Plotting the Candlestick Chart ---
    # We use the mplfinance.plot() function, which is specifically designed for this.
    # The tutorial provides a good foundation for its parameters.
    mpf.plot(
        resampled_df,
        type='candle',  # Specifies the chart type as candlestick.
        style='yahoo',  # Sets a classic visual style for the chart.
        title=f'{ticker} Candlestick Chart ({n_days}-Day Period)', # Dynamic title.
        ylabel='Price ($)', # Label for the y-axis.
        volume=False # We are not showing the volume panel for this chart.
    )

def plot_price_distribution_boxplot(df, ticker, n_days=21):
    """
    Displays a boxplot showing the distribution of the 'Close' price
    over a rolling window of 'n' trading days.

    This helps to visualize price volatility and trends over specific periods.

    Args:
        df (pd.DataFrame): The input DataFrame containing the stock data.
            It must have a 'Close' column.
        ticker (str): The stock ticker symbol (e.g., "AAPL") for the chart title.
        n_days (int): The number of trading days in the rolling window.
            Default is 21, which is approximately one trading month.
    """

    # --- Requirement: Display data for a moving window of n days ---
    # To create a boxplot for a moving window, we first need to reshape our data.
    # We will create a new DataFrame where each column represents a day in the
    # n-day window, and each row is a separate window.

    # Extract the 'Close' price series from the main DataFrame.
    close_prices = df['Close']

    # Calculate how many full n-day windows we can create from the data.
    num_windows = len(close_prices) // n_days

    # Trim the data to only include full windows.
    trimmed_prices = close_prices[:num_windows * n_days]

    # Reshape the 1D series of prices into a 2D array.
    # The shape will be (number_of_windows, n_days_per_window).
    # For example, with 100 days and n_days=20, this creates an array of 5 rows and 20 columns.
    reshaped_prices = trimmed_prices.values.reshape(num_windows, n_days)

    # Convert this 2D array back into a DataFrame for easy plotting.
    boxplot_df = pd.DataFrame(reshaped_prices)

    # --- Plotting the Boxplot Chart ---
    plt.figure(figsize=(16, 8))

    # Create the boxplot from our reshaped DataFrame.
    boxplot_df.T.boxplot() # We transpose (.T) so each window is a box.

    plt.title(f'{ticker} Price Distribution ({n_days}-Day Rolling Window)')
    plt.xlabel('Window Period')
    plt.ylabel('Price ($)')
    plt.xticks(rotation=45) # Rotate x-axis labels for better readability.
    plt.grid(True)
    plt.show()