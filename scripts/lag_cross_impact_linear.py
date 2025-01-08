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
lagged_linear_folder = os.path.join(output_folder, 'lagged_linear')  # Subfolder for lagged linear analysis
regression_results_folder = os.path.join(lagged_linear_folder, 'regression_results')  # Subfolder for regression results
plots_folder = os.path.join(lagged_linear_folder, 'plots')  # Subfolder for plots
self_vs_cross_folder = os.path.join(lagged_linear_folder, 'self_vs_cross_impact')  # Subfolder for self vs. cross-impact

# Create directories if they don't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(lagged_linear_folder, exist_ok=True)
os.makedirs(regression_results_folder, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)
os.makedirs(self_vs_cross_folder, exist_ok=True)

def initialize_dask_client():
    """
    Initialize a Dask client with a LocalCluster.
    Adjust the number of workers and memory limits based on your VM's resources.
    """
    cluster = LocalCluster(n_workers=8, threads_per_worker=2, memory_limit='16GB')  # Adjust as needed
    client = Client(cluster)
    return client

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

def process_chunk(chunk):
    """
    Process a chunk of data:
    - Remove duplicates.
    - Resample to a regular time series.
    - Compute mid-price and price changes.
    """
    # Remove duplicate timestamps
    chunk = chunk[~chunk.index.duplicated(keep='first')]

    # Resample to a regular time series (e.g., 100ms intervals)
    chunk = chunk.resample('100ms').ffill()

    # Compute mid-price and price changes
    chunk['mid_price'] = (chunk['bid_px_00'] + chunk['ask_px_00']) / 2
    chunk['price_change'] = chunk.groupby('symbol')['mid_price'].diff()

    # Reset the index to make 'ts_recv' a column again
    chunk = chunk.reset_index()

    # Fill NaN values
    chunk['price_change'] = chunk['price_change'].fillna(0)

    return chunk

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

def self_impact(data, stock, lag='50ms'):
    """
    Analyze how the OFI of a stock affects its own price changes at a future time horizon.
    """
    # Filter data for the stock
    stock_data = data[data['symbol'] == stock][['ts_recv', 'ofi_pca', 'price_change']]

    # Shift OFI values by the specified lag
    stock_data['ts_recv'] = stock_data['ts_recv'] + pd.Timedelta(lag)
    stock_data = stock_data.rename(columns={'ofi_pca': 'ofi_pca_lagged'})

    # Drop rows with NaN values
    stock_data = stock_data.dropna()

    # Check if there is sufficient variation in the data
    if stock_data['price_change'].nunique() < 2 or stock_data['ofi_pca_lagged'].nunique() < 2:
        print(f"Warning: Insufficient variation in data for {stock} → {stock}. Skipping regression.")
        return None

    # Perform regression
    X = sm.add_constant(stock_data['ofi_pca_lagged'])
    y = stock_data['price_change']
    model = sm.OLS(y, X).fit()

    return model

def compare_self_vs_cross_impact(data, stock_a, stock_b, lag='50ms'):
    """
    Compare self-impact (stock_a → stock_a) vs. cross-impact (stock_a → stock_b).
    """
    # Analyze self-impact
    self_model = self_impact(data, stock_a, lag=lag)
    if self_model is not None:
        self_r_squared = self_model.rsquared
        print(f"Self-Impact R-squared ({stock_a} → {stock_a}): {self_r_squared}")
    else:
        self_r_squared = None

    # Analyze cross-impact
    cross_model = lagged_cross_impact(data, stock_a, stock_b, lag=lag)
    if cross_model is not None:
        cross_r_squared = cross_model.rsquared
        print(f"Cross-Impact R-squared ({stock_a} → {stock_b}): {cross_r_squared}")
    else:
        cross_r_squared = None

    # Save comparison results
    if self_r_squared is not None and cross_r_squared is not None:
        with open(os.path.join(self_vs_cross_folder, f'self_vs_cross_impact_{stock_a}_to_{stock_b}_{lag}.txt'), 'w') as f:
            f.write(f"Self-Impact R-squared ({stock_a} → {stock_a}): {self_r_squared}\n")
            f.write(f"Cross-Impact R-squared ({stock_a} → {stock_b}): {cross_r_squared}\n")

        # Visualize comparison
        plt.figure(figsize=(8, 6))
        plt.bar(['Self-Impact', 'Cross-Impact'], [self_r_squared, cross_r_squared], color=['blue', 'orange'])
        plt.title(f'Self-Impact vs. Cross-Impact: {stock_a} → {stock_b} (lag={lag})')
        plt.ylabel('R-squared')
        plt.grid(True)
        plt.savefig(os.path.join(self_vs_cross_folder, f'self_vs_cross_impact_{stock_a}_to_{stock_b}_{lag}.png'))
        plt.close()

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

        # Analyze lagged cross-impact for all pairs of stocks
        for stock_a in stocks:
            for stock_b in stocks:
                if stock_a != stock_b:
                    # Compare self-impact vs. cross-impact for each lag
                    for lag in lags_to_test:
                        print(f"Comparing self-impact vs. cross-impact: {stock_a} → {stock_b} (lag={lag})")
                        compare_self_vs_cross_impact(data, stock_a, stock_b, lag=lag)

        print("Analysis complete. Results saved to:", output_folder)

    except KeyboardInterrupt:
        print("Script interrupted by user.")
    finally:
        # Shut down the Dask client
        client.close()
        print("Dask client shut down.")

if __name__ == "__main__":
    main()
