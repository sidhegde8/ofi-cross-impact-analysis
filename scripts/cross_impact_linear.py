import dask.dataframe as dd
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from dask.distributed import Client, LocalCluster
from multiprocessing import freeze_support
import matplotlib.pyplot as plt
import seaborn as sns

# Set up directories
input_folder = 'data/parquet'  # Update this path to your input folder
output_folder = 'results'      # Update this path to your output folder
os.makedirs(output_folder, exist_ok=True)

# Initialize Dask client
def initialize_dask_client():
    """
    Initialize a Dask client with a LocalCluster.
    Adjust the number of workers and memory limits based on your VM's resources.
    """
    cluster = LocalCluster(n_workers=8, threads_per_worker=2, memory_limit='16GB')  # Adjust as needed
    client = Client(cluster)
    return client

# Load processed data using Dask
def load_processed_data(input_folder):
    """
    Load all processed Parquet files from the input folder using Dask.
    """
    processed_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.parquet')]
    print("Parquet files found:", processed_files)

    if not processed_files:
        raise FileNotFoundError("No Parquet files found in the directory.")

    # Read Parquet files
    data = dd.read_parquet(processed_files)

    # Print column names for debugging
    print("Columns in the DataFrame:", data.columns.tolist())

    # Clean column names (remove leading/trailing spaces and convert to lowercase)
    data.columns = data.columns.str.strip().str.lower()

    # Verify 'ts_recv' column exists
    if 'ts_recv' not in data.columns:
        raise KeyError("'ts_recv' column not found in DataFrame.")

    # Ensure 'ts_recv' is a datetime column
    data['ts_recv'] = dd.to_datetime(data['ts_recv'])

    # Set 'ts_recv' as the index
    data = data.set_index('ts_recv', sorted=True).persist()

    return data

# Process data in chunks
def process_chunk(chunk):
    """
    Process a chunk of data:
    - Remove duplicates.
    - Resample to a regular time series.
    - Compute mid-price and price changes.
    """
    # Print column names for debugging
    print("Columns in the chunk before processing:", chunk.columns.tolist())

    # Remove duplicate timestamps
    chunk = chunk[~chunk.index.duplicated(keep='first')]

    # Resample to a regular time series (e.g., 100ms intervals)
    chunk = chunk.resample('100ms').ffill()

    # Compute mid-price and price changes
    chunk['mid_price'] = (chunk['bid_px_00'] + chunk['ask_px_00']) / 2
    chunk['price_change'] = chunk.groupby('symbol')['mid_price'].diff()

    # Reset the index to make 'ts_recv' a column again
    chunk = chunk.reset_index()

    # Print column names for debugging
    print("Columns in the chunk after processing:", chunk.columns.tolist())

    # Verify that 'price_change' does not contain unexpected NaN values
    if chunk['price_change'].isna().any():
        print("Warning: NaN values found in 'price_change'. Filling with 0.")
        chunk['price_change'] = chunk['price_change'].fillna(0)

    return chunk

# Analyze lagged cross-impact
def lagged_cross_impact(data, stock_a, stock_b, lag='50ms'):
    """
    Analyze how the OFI of one stock affects the price changes of another stock at a future time horizon.
    """
    # Filter data for the two stocks
    stock_a_data = data[data['symbol'] == stock_a][['ts_recv', 'ofi_pca']]
    stock_b_data = data[data['symbol'] == stock_b][['ts_recv', 'price_change']]

    # Shift OFI values by the specified lag
    stock_a_data['ts_recv'] = stock_a_data['ts_recv'] + pd.Timedelta(lag)
    stock_a_data = stock_a_data.rename(columns={'ofi_pca': 'ofi_pca_lagged'})

    # Merge the two datasets on the nearest timestamp
    merged_data = pd.merge_asof(
        stock_a_data.sort_values('ts_recv'),
        stock_b_data.sort_values('ts_recv'),
        on='ts_recv',
        direction='nearest'
    )

    # Drop rows with NaN values in 'price_change' or 'ofi_pca_lagged'
    merged_data = merged_data.dropna(subset=['price_change', 'ofi_pca_lagged'])

    # Check if merged_data has at least 2 rows
    if len(merged_data) < 2:
        print(f"Warning: Insufficient overlapping timestamps after applying lag for {stock_a} -> {stock_b}. Skipping regression.")
        return None

    # Check for sufficient variation in the data
    if merged_data['price_change'].nunique() < 2 or merged_data['ofi_pca_lagged'].nunique() < 2:
        print(f"Warning: Insufficient variation in 'price_change' or 'ofi_pca_lagged' for {stock_a} -> {stock_b}. Skipping regression.")
        return None

    # Perform regression
    X = sm.add_constant(merged_data['ofi_pca_lagged'])
    y = merged_data['price_change']
    model = sm.OLS(y, X).fit()

    return model

# Save regression results
def save_regression_results(model, filename):
    """
    Save regression results to a text file.
    """
    if model is None:
        print(f"Warning: No model to save for {filename}.")
        return

    with open(os.path.join(output_folder, filename), 'w') as f:
        f.write(model.summary().as_text())

# Visualize R-squared vs. Lag
def visualize_r_squared(lags, r_squared_values, stock_a, stock_b):
    """
    Plot R-squared values for different lags.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(lags, r_squared_values, marker='o')
    plt.title(f'R-squared vs. Lag for {stock_a} → {stock_b}')
    plt.xlabel('Lag Interval')
    plt.ylabel('R-squared')
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, f'r_squared_vs_lag_{stock_a}_to_{stock_b}.png'))
    plt.close()

# Main function
def main():
    freeze_support()  # Required for Windows/macOS

    # Initialize Dask client
    client = initialize_dask_client()

    try:
        # List of stocks to analyze
        stocks = ['AAPL', 'AMGN', 'TSLA', 'JPM', 'XOM']

        # Load and process data using Dask
        print("Loading data using Dask...")
        data = load_processed_data(input_folder)

        # Process data in chunks
        print("Processing data in chunks...")
        data = data.map_partitions(process_chunk).compute()

        # Define a list of lags to test
        lags_to_test = ['50ms', '100ms', '500ms', '1s', '5s', '10s', '1min', '5min']

        # Dictionary to store the best lag for each stock pair
        best_lags = {}

        # Analyze lagged cross-impact for all pairs of stocks
        for stock_a in stocks:
            for stock_b in stocks:
                if stock_a != stock_b:
                    r_squared_values = []
                    closest_to_one = float('inf')  # Initialize with a large value
                    best_lag = None
                    best_model = None

                    for lag in lags_to_test:
                        print(f"Analyzing lagged cross-impact: {stock_a} -> {stock_b} (lag={lag})")
                        model = lagged_cross_impact(data, stock_a, stock_b, lag=lag)
                        if model is not None:
                            r_squared = model.rsquared
                            r_squared_values.append(r_squared)
                            # Debugging: Print R-squared and its difference from 1
                            print(f"R-squared for {lag}: {r_squared}, Difference from 1: {abs(1 - r_squared)}")
                            # Track the lag with R-squared closest to 1
                            if abs(1 - r_squared) < closest_to_one:
                                closest_to_one = abs(1 - r_squared)
                                best_lag = lag
                                best_model = model
                                # Debugging: Print the current best lag and its R-squared
                                print(f"New best lag: {best_lag}, R-squared: {r_squared}, Difference from 1: {closest_to_one}")

                    # Save the best model for this stock pair
                    if best_model is not None:
                        save_regression_results(best_model, f'best_lagged_{stock_a}_to_{stock_b}_{best_lag}.txt')
                        best_lags[f'{stock_a}_to_{stock_b}'] = best_lag

                    # Visualize R-squared vs. Lag for this stock pair
                    visualize_r_squared(lags_to_test, r_squared_values, stock_a, stock_b)

        # Print the best lags for each stock pair
        print("Best lags for each stock pair (R-squared closest to 1):")
        for pair, lag in best_lags.items():
            print(f"{pair}: {lag}")

        print("Analysis complete. Results saved to:", output_folder)

    except KeyboardInterrupt:
        print("Script interrupted by user.")
    finally:
        # Shut down the Dask client
        client.close()
        print("Dask client shut down.")

if __name__ == "__main__":
    main()