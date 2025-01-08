import dask.dataframe as dd
import pandas as pd
import numpy as np
import os
from dask.distributed import Client, LocalCluster
from multiprocessing import freeze_support
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Set up directories
input_folder = 'data/parquet'  # Update this path to your input folder
output_folder = 'results'      # Update this path to your output folder

# Create subfolders for organized results
subfolders = [
    'model_comparison',
    'best_lagged_results',
    'r_squared_plots'
]
for folder in subfolders:
    os.makedirs(os.path.join(output_folder, folder), exist_ok=True)

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
    - Compute mid-price, price changes, volume, and volatility.
    """
    # Remove duplicate timestamps
    chunk = chunk[~chunk.index.duplicated(keep='first')]

    # Resample to a regular time series (e.g., 100ms intervals)
    chunk = chunk.resample('100ms').ffill()

    # Compute mid-price and price changes
    chunk['mid_price'] = (chunk['bid_px_00'] + chunk['ask_px_00']) / 2
    chunk['price_change'] = chunk.groupby('symbol')['mid_price'].diff()

    # Compute volume (bid + ask size)
    chunk['volume'] = chunk['bid_sz_00'] + chunk['ask_sz_00']

    # Compute volatility (rolling standard deviation of mid-price changes)
    chunk['volatility'] = chunk.groupby('symbol')['mid_price'].diff().rolling(window=10).std()

    # Reset the index to make 'ts_recv' a column again
    chunk = chunk.reset_index()

    # Fill NaN values
    chunk['price_change'] = chunk['price_change'].fillna(0)
    chunk['volatility'] = chunk['volatility'].fillna(0)

    return chunk

# Analyze lagged cross-impact using Random Forest
def lagged_cross_impact(data, stock_a, stock_b, lag='50ms', use_additional_features=True):
    """
    Analyze how the OFI of one stock affects the price changes of another stock at a future time horizon.
    """
    # Filter data for the two stocks
    stock_a_data = data[data['symbol'] == stock_a][['ts_recv', 'ofi_pca', 'volume', 'volatility']]
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

    # Drop rows with NaN values
    merged_data = merged_data.dropna()

    # Features and target
    if use_additional_features:
        X = merged_data[['ofi_pca_lagged', 'volume', 'volatility']]  # Include additional features
    else:
        X = merged_data[['ofi_pca_lagged']]  # Only OFI

    y = merged_data['price_change']

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=75, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    # Calculate R-squared on the validation set
    r_squared = r2_score(y_val, model.predict(X_val))
    print(f"R-squared for {lag} (validation set): {r_squared}")

    return model, r_squared

# Save regression results
def save_regression_results(model, r_squared, filename, folder):
    """
    Save regression results to a text file, including feature importances and R-squared.
    """
    if model is None:
        print(f"Warning: No model to save for {filename}.")
        return

    with open(os.path.join(output_folder, folder, filename), 'w') as f:
        f.write(f"Random Forest Model Summary:\n")
        f.write(f"R-squared: {r_squared}\n")
        f.write("Feature Importances:\n")
        for feature, importance in zip(model.feature_names_in_, model.feature_importances_):
            f.write(f"{feature}: {importance}\n")

# Visualize R-squared vs. Lag
def visualize_r_squared(lags, baseline_r_squared, enhanced_r_squared, stock_a, stock_b):
    """
    Plot R-squared values for different lags.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(lags, baseline_r_squared, marker='o', label='Baseline (OFI Only)')
    plt.plot(lags, enhanced_r_squared, marker='o', label='Enhanced (OFI + Features)')
    plt.title(f'R-squared vs. Lag for {stock_a} → {stock_b}')
    plt.xlabel('Lag Interval')
    plt.ylabel('R-squared')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'r_squared_plots', f'r_squared_vs_lag_{stock_a}_to_{stock_b}.png'))
    plt.close()

# Main function
def main():
    freeze_support()
    client = initialize_dask_client()

    try:
        stocks = ['AAPL', 'AMGN', 'TSLA', 'JPM', 'XOM']
        print("Loading data using Dask...")
        data = load_processed_data(input_folder)

        print("Processing data in chunks...")
        data = data.map_partitions(process_chunk).compute()

        lags_to_test = ['50ms', '100ms', '500ms', '1s', '5s', '10s', '1min', '5min']
        best_lags = {}

        for stock_a in stocks:
            for stock_b in stocks:
                if stock_a != stock_b:
                    # Initialize variables at the start
                    baseline_r_squared_values = []
                    enhanced_r_squared_values = []
                    valid_lags = []
                    closest_to_one = float('inf')
                    best_lag = None
                    best_model = None
                    best_r_squared = None

                    # Track if we found any valid models
                    found_valid_model = False

                    # Dictionary to store R-squared values for each lag
                    lag_r_squared = {}

                    for lag in lags_to_test:
                        print(f"Analyzing lagged cross-impact: {stock_a} -> {stock_b} (lag={lag})")
                        
                        # Train baseline model (only OFI)
                        print("Training baseline model (only OFI)...")
                        baseline_model, baseline_r_squared = lagged_cross_impact(data, stock_a, stock_b, lag=lag, use_additional_features=False)
                        baseline_r_squared_values.append(baseline_r_squared)
                        print(f"Baseline R-squared for {lag}: {baseline_r_squared}")

                        # Train enhanced model (OFI + volume + volatility)
                        print("Training enhanced model (OFI + volume + volatility)...")
                        enhanced_model, enhanced_r_squared = lagged_cross_impact(data, stock_a, stock_b, lag=lag, use_additional_features=True)
                        enhanced_r_squared_values.append(enhanced_r_squared)
                        print(f"Enhanced R-squared for {lag}: {enhanced_r_squared}")

                        # Store R-squared for this lag
                        lag_r_squared[lag] = {
                            'baseline': baseline_r_squared,
                            'enhanced': enhanced_r_squared
                        }

                        # Check if this is the best lag
                        if abs(1 - enhanced_r_squared) < closest_to_one:
                            closest_to_one = abs(1 - enhanced_r_squared)
                            best_lag = lag
                            best_model = enhanced_model
                            best_r_squared = enhanced_r_squared
                            print(f"New best lag: {best_lag}, R-squared: {best_r_squared}, Difference from 1: {closest_to_one}")

                    # Only try to save results if we found at least one valid model
                    if best_model is not None and best_r_squared is not None and best_lag is not None:
                        try:
                            # Save best model results
                            save_regression_results(
                                best_model, 
                                best_r_squared, 
                                f'best_lagged_{stock_a}_to_{stock_b}_{best_lag}.txt',
                                'best_lagged_results'
                            )

                            # Save R-squared values for all lags
                            with open(os.path.join(output_folder, 'model_comparison', f'r_squared_values_{stock_a}_to_{stock_b}.txt'), 'w') as f:
                                f.write("R-squared values for each lag:\n")
                                for lag, r_sq in lag_r_squared.items():
                                    f.write(f"{lag}:\n")
                                    f.write(f"  Baseline (OFI Only): {r_sq['baseline']}\n")
                                    f.write(f"  Enhanced (OFI + Features): {r_sq['enhanced']}\n")

                            # Visualize R-squared vs. Lag
                            visualize_r_squared(lags_to_test, baseline_r_squared_values, enhanced_r_squared_values, stock_a, stock_b)
                        except Exception as e:
                            print(f"Error saving results for {stock_a} -> {stock_b}: {str(e)}")
                    else:
                        print(f"Warning: No valid model found for {stock_a} -> {stock_b}")

        if best_lags:
            print("\nBest lags for each stock pair (R-squared closest to 1):")
            for pair, lag in best_lags.items():
                print(f"{pair}: {lag}")
        else:
            print("\nNo valid models found for any stock pairs")

        print("Analysis complete. Results saved to:", output_folder)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        client.close()
        print("Dask client shut down.")

if __name__ == "__main__":
    main()