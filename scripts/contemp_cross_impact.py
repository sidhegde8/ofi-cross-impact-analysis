import dask.dataframe as dd
import pandas as pd
import numpy as np
import os
from dask.distributed import Client, LocalCluster
from multiprocessing import freeze_support
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Set up directories
input_folder = 'data/parquet'
output_folder = 'results/contemporaneous_cross_impact'

# Create subfolder for generalized results
generalized_results_folder = os.path.join(output_folder, 'generalized_results')
os.makedirs(generalized_results_folder, exist_ok=True)

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

# Analyze self-impact using Random Forest
def self_impact(data, stock):
    """
    Analyze the self-impact of a stock's OFI on its own price changes using Random Forest.
    """
    # Filter data for the stock
    stock_data = data[data['symbol'] == stock][['ts_recv', 'ofi_pca', 'volume', 'volatility', 'price_change']]

    # Drop rows with NaN values
    stock_data = stock_data.dropna()

    # Features and target
    X = stock_data[['ofi_pca', 'volume', 'volatility']]  # Independent variables: OFI, volume, volatility
    y = stock_data['price_change']  # Dependent variable: Price change

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=75, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    # Calculate R-squared on the validation set
    r_squared = r2_score(y_val, model.predict(X_val))
    print(f"Self-impact R-squared for {stock}: {r_squared}")

    # Extract feature importance
    feature_importance = model.feature_importances_[0]  # Importance of OFI_PCA

    return feature_importance, r_squared

# Analyze cross-impact using Random Forest
def cross_impact(data, stock_a, stock_b):
    """
    Analyze the cross-impact of one stock's OFI on another stock's price changes using Random Forest.
    """
    # Filter data for the two stocks
    stock_a_data = data[data['symbol'] == stock_a][['ts_recv', 'ofi_pca', 'volume', 'volatility']]
    stock_b_data = data[data['symbol'] == stock_b][['ts_recv', 'price_change']]

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
    X = merged_data[['ofi_pca', 'volume', 'volatility']]  # Independent variables: OFI, volume, volatility
    y = merged_data['price_change']  # Dependent variable: Price change of stock_b

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=75, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    # Calculate R-squared on the validation set
    r_squared = r2_score(y_val, model.predict(X_val))
    print(f"Cross-impact R-squared for {stock_a} → {stock_b}: {r_squared}")

    # Extract feature importance
    feature_importance = model.feature_importances_[0]  # Importance of OFI_PCA

    return feature_importance, r_squared

# Save generalized results
def save_generalized_results(results_df, generalized_results_folder):
    """
    Save generalized results to a CSV file.
    """
    csv_filename = os.path.join(generalized_results_folder, 'generalized_results.csv')
    results_df.to_csv(csv_filename, index=False)
    print(f"Generalized results saved to: {csv_filename}")

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

        results = []

        # Analyze self-impact for each stock
        for stock in stocks:
            print(f"Analyzing self-impact for {stock}...")
            try:
                feature_importance, r_squared = self_impact(data, stock)
                results.append((stock, stock, feature_importance, r_squared, 'Self-Impact'))
            except Exception as e:
                print(f"Error analyzing self-impact for {stock}: {str(e)}")

        # Analyze cross-impact for all stock pairs
        for stock_a in stocks:
            for stock_b in stocks:
                if stock_a != stock_b:
                    print(f"Analyzing cross-impact: {stock_a} → {stock_b}")
                    try:
                        feature_importance, r_squared = cross_impact(data, stock_a, stock_b)
                        results.append((stock_a, stock_b, feature_importance, r_squared, 'Cross-Impact'))
                    except Exception as e:
                        print(f"Error analyzing cross-impact for {stock_a} → {stock_b}: {str(e)}")

        # Convert results to a DataFrame
        results_df = pd.DataFrame(results, columns=['Stock_A', 'Stock_B', 'Feature_Importance', 'R_Squared', 'Impact_Type'])

        # Save generalized results
        save_generalized_results(results_df, generalized_results_folder)

        print("Contemporaneous cross-impact analysis complete. Results saved to:", generalized_results_folder)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        client.close()
        print("Dask client shut down.")

if __name__ == "__main__":
    main()
