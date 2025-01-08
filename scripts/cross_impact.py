import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import dask.dataframe as dd

# Set up directories
input_folder = 'data/processed'
output_folder = 'results'
os.makedirs(output_folder, exist_ok=True)

# Define sectors
sectors = {
    'Tech': ['AAPL'],
    'Healthcare': ['AMGN'],
    'Consumer Discretionary': ['TSLA'],
    'Financials': ['JPM'],
    'Energy': ['XOM']
}

# Load all processed files using Dask
def load_processed_data(input_folder):
    """
    Load all processed CSV files from the input folder using Dask.
    """
    processed_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.startswith('processed_')]
    
    # Load data using Dask
    data = dd.read_csv(processed_files, parse_dates=['ts_recv'])
    
    # Sort by timestamp
    data = data.set_index('ts_recv').persist()
    
    return data

# Interpolate missing timestamps using Dask
def interpolate_data(data, freq='5S'):
    """
    Interpolate missing timestamps to create a regular time series using Dask.
    """
    # Convert Dask DataFrame to pandas DataFrame for resampling
    data = data.compute()
    
    # Check for duplicate timestamps and remove them
    if data.index.duplicated().any():
        print("Duplicate timestamps found. Removing duplicates...")
        data = data[~data.index.duplicated(keep='first')]
    
    # Resample the data
    data_resampled = data.groupby('symbol').apply(lambda group: group.resample(freq).ffill()).reset_index(drop=True)
    
    return data_resampled

# Compute mid-price and price changes
def compute_price_changes(data):
    """
    Compute mid-price and price changes for each stock.
    """
    # Calculate mid-price
    data['mid_price'] = (data['bid_px_00'] + data['ask_px_00']) / 2
    
    # Calculate price change (difference in mid-price)
    data['price_change'] = data.groupby('symbol')['mid_price'].diff()
    
    return data

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
    
    # Merge the two datasets on timestamp
    merged_data = pd.merge(stock_a_data, stock_b_data, on='ts_recv', how='inner')
    
    # Check if merged_data has at least 2 rows
    if len(merged_data) < 2:
        print(f"Warning: Insufficient overlapping timestamps after applying lag for {stock_a} -> {stock_b}. Skipping regression.")
        return None
    
    # Perform regression
    X = sm.add_constant(merged_data['ofi_pca_lagged'])
    y = merged_data['price_change']
    model = sm.OLS(y, X).fit()
    
    return model

# Analyze lagged cross-impact for all pairs of stocks
def lagged_cross_impact_all_pairs(data, stocks, lag='50ms'):
    """
    Analyze lagged cross-impact for all pairs of stocks.
    """
    results = {}
    
    # Loop through all pairs of stocks
    for stock_a in stocks:
        for stock_b in stocks:
            if stock_a != stock_b:
                print(f"Analyzing lagged cross-impact: {stock_a} -> {stock_b} (lag={lag})")
                model = lagged_cross_impact(data, stock_a, stock_b, lag=lag)
                if model is not None:  # Only save results if the model is not None
                    results[f'{stock_a}_to_{stock_b}'] = model
                    save_regression_results(model, f'lagged_{stock_a}_to_{stock_b}.txt')
    
    return results

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

# Main function
def main():
    # List of stocks to analyze
    stocks = ['AAPL', 'AMGN', 'TSLA', 'JPM', 'XOM']
    
    # Load and process data using Dask
    print("Loading data using Dask...")
    data = load_processed_data(input_folder)
    
    # Interpolate missing timestamps
    print("Interpolating missing timestamps...")
    data = interpolate_data(data, freq='5S')  # Convert to pandas DataFrame for resampling
    
    # Compute mid-price and price changes
    print("Computing price changes...")
    data = compute_price_changes(data)
    
    # Analyze lagged cross-impact for all pairs of stocks
    print("Analyzing lagged cross-impact...")
    lagged_cross_impact_all_pairs(data, stocks, lag='50ms')

if __name__ == "__main__":
    main()