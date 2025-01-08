import os
import pandas as pd
import dask.dataframe as dd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from dask.distributed import Client, LocalCluster
from multiprocessing import freeze_support

# Set up directories
input_folder = 'data/parquet'  # Folder containing processed Parquet files
output_folder = 'results/contemporaneous_cross_impact'  # Main folder for results

# Create subfolders for organized results
subfolders = [
    'regression_results',  # For regression results (text files)
    'visualizations',      # For scatter plots and visualizations
    'summary_statistics',  # For aggregated results or summary statistics
    'self_vs_cross_impact' # For self-impact vs. cross-impact comparisons
]

for folder in subfolders:
    os.makedirs(os.path.join(output_folder, folder), exist_ok=True)

def initialize_dask_client():
    """
    Initialize a Dask client with a LocalCluster.
    Adjust the number of workers and memory limits based on your VM's resources.
    """
    cluster = LocalCluster(
        n_workers=8,  # Adjust based on your system's resources
        threads_per_worker=2,  # Adjust based on your CPU cores
        memory_limit='16GB',  # Adjust based on your system's available RAM
        processes=True  # Use separate processes for each worker
    )
    client = Client(cluster)
    return client

def load_processed_data(input_folder):
    """
    Load processed Parquet files from the input folder using Dask.
    """
    processed_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.parquet')]
    print("Parquet files found:", processed_files)

    if not processed_files:
        raise FileNotFoundError("No Parquet files found in the directory.")

    # Read Parquet files using Dask
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

def contemporaneous_cross_impact(data, stock_a, stock_b):
    """
    Analyze the contemporaneous cross-impact of OFI on price changes.
    """
    # Filter data for the two stocks
    stock_a_data = data[data['symbol'] == stock_a][['ts_recv', 'ofi_pca']]
    stock_b_data = data[data['symbol'] == stock_b][['ts_recv', 'price_change']]

    # Merge the two datasets on the nearest timestamp using pd.merge_asof
    merged_data = pd.merge_asof(
        stock_a_data.sort_values('ts_recv'),
        stock_b_data.sort_values('ts_recv'),
        on='ts_recv',
        direction='nearest'
    )

    # Drop rows with NaN values
    merged_data = merged_data.dropna()

    # Check if merged_data is empty
    if merged_data.empty:
        print(f"Warning: No overlapping data found for {stock_a} → {stock_b}. Skipping regression.")
        return None, None

    # Check if there is sufficient variation in the data
    if merged_data['price_change'].nunique() < 2 or merged_data['ofi_pca'].nunique() < 2:
        print(f"Warning: Insufficient variation in data for {stock_a} → {stock_b}. Skipping regression.")
        return None, None

    # Perform regression
    X = sm.add_constant(merged_data['ofi_pca'])  # Independent variable: OFI of stock_a
    y = merged_data['price_change']  # Dependent variable: Price change of stock_b
    model = sm.OLS(y, X).fit()

    return model, merged_data

def self_impact(data, stock):
    """
    Analyze the contemporaneous self-impact of OFI on price changes.
    """
    # Filter data for the stock
    stock_data = data[data['symbol'] == stock][['ts_recv', 'ofi_pca', 'price_change']]

    # Drop rows with NaN values
    stock_data = stock_data.dropna()

    # Check if there is sufficient variation in the data
    if stock_data['price_change'].nunique() < 2 or stock_data['ofi_pca'].nunique() < 2:
        print(f"Warning: Insufficient variation in data for {stock} → {stock}. Skipping regression.")
        return None, None

    # Perform regression
    X = sm.add_constant(stock_data['ofi_pca'])  # Independent variable: OFI of stock
    y = stock_data['price_change']  # Dependent variable: Price change of stock
    model = sm.OLS(y, X).fit()

    return model, stock_data

def compare_self_vs_cross_impact(data, stock_a, stock_b):
    """
    Compare self-impact (stock_a → stock_a) vs. cross-impact (stock_a → stock_b).
    """
    # Analyze self-impact
    self_model, self_data = self_impact(data, stock_a)
    if self_model is not None:
        self_r_squared = self_model.rsquared
        print(f"Self-Impact R-squared ({stock_a} → {stock_a}): {self_r_squared}")
    else:
        self_r_squared = None

    # Analyze cross-impact
    cross_model, cross_data = contemporaneous_cross_impact(data, stock_a, stock_b)
    if cross_model is not None:
        cross_r_squared = cross_model.rsquared
        print(f"Cross-Impact R-squared ({stock_a} → {stock_b}): {cross_r_squared}")
    else:
        cross_r_squared = None

    # Save comparison results
    if self_r_squared is not None and cross_r_squared is not None:
        with open(os.path.join(output_folder, 'self_vs_cross_impact', f'self_vs_cross_impact_{stock_a}_to_{stock_b}.txt'), 'w') as f:
            f.write(f"Self-Impact R-squared ({stock_a} → {stock_a}): {self_r_squared}\n")
            f.write(f"Cross-Impact R-squared ({stock_a} → {stock_b}): {cross_r_squared}\n")

        # Visualize comparison
        plt.figure(figsize=(8, 6))
        plt.bar(['Self-Impact', 'Cross-Impact'], [self_r_squared, cross_r_squared], color=['blue', 'orange'])
        plt.title(f'Self-Impact vs. Cross-Impact: {stock_a} → {stock_b}')
        plt.ylabel('R-squared')
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, 'self_vs_cross_impact', f'self_vs_cross_impact_{stock_a}_to_{stock_b}.png'))
        plt.close()

def main():
    freeze_support()  # Required for Windows/macOS

    # Initialize Dask client
    client = initialize_dask_client()

    try:
        # Load processed data
        print("Loading processed data...")
        data = load_processed_data(input_folder)

        # Process data in chunks
        print("Processing data in chunks...")
        data = data.map_partitions(process_chunk).compute()

        # List of stocks to analyze
        stocks = ['AAPL', 'AMGN', 'TSLA', 'JPM', 'XOM']

        # Analyze contemporaneous cross-impact for all pairs of stocks
        for stock_a in stocks:
            for stock_b in stocks:
                if stock_a != stock_b:
                    print(f"Analyzing contemporaneous cross-impact: {stock_a} → {stock_b}")
                    try:
                        # Perform regression analysis
                        model, merged_data = contemporaneous_cross_impact(data, stock_a, stock_b)
                        
                        if model is not None and merged_data is not None:
                            # Save regression results
                            save_regression_results(model, stock_a, stock_b, output_folder)
                            
                            # Save summary statistics
                            save_summary_statistics(merged_data, stock_a, stock_b, output_folder)
                            
                            # Visualize the relationship
                            visualize_contemporaneous_impact(model, merged_data, stock_a, stock_b, output_folder)
                            
                            print(f"Results saved for {stock_a} → {stock_b}")
                        else:
                            print(f"Skipping {stock_a} → {stock_b} due to insufficient data.")
                    except Exception as e:
                        print(f"Error analyzing {stock_a} → {stock_b}: {str(e)}")

        # Compare self-impact vs. cross-impact for all pairs of stocks
        for stock_a in stocks:
            for stock_b in stocks:
                if stock_a != stock_b:
                    print(f"Comparing self-impact vs. cross-impact: {stock_a} → {stock_b}")
                    compare_self_vs_cross_impact(data, stock_a, stock_b)

        print("Contemporaneous cross-impact analysis complete. Results saved to:", output_folder)

    except KeyboardInterrupt:
        print("Script interrupted by user.")
    finally:
        # Shut down the Dask client
        client.close()
        print("Dask client shut down.")

if __name__ == "__main__":
    main()
