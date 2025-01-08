import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA

def load_cleaned_data(input_folder):
    """
    Load all cleaned CSV files from the input folder.
    Replace empty values (e.g., "") with NaN.
    """
    cleaned_files = [f for f in os.listdir(input_folder) if f.startswith('cleaned_')]
    dataframes = []
    
    for file in cleaned_files:
        file_path = os.path.join(input_folder, file)
        
        # Load the CSV, treating empty strings as NaN
        df = pd.read_csv(file_path, na_values=[''])
        dataframes.append(df)
    
    # Combine all dataframes into one
    combined_data = pd.concat(dataframes, ignore_index=True)
    return combined_data

def calculate_ofi(data):
    """
    Calculate OFI for each level of the limit order book.
    """
    for level in range(5):
        # Calculate changes in bid and ask sizes
        data[f'bid_sz_{level:02d}_diff'] = data[f'bid_sz_{level:02d}'].diff()
        data[f'ask_sz_{level:02d}_diff'] = data[f'ask_sz_{level:02d}'].diff()
        
        # Compute OFI for each level
        data[f'ofi_{level:02d}'] = data[f'bid_sz_{level:02d}_diff'] - data[f'ask_sz_{level:02d}_diff']
    
    return data

def integrate_ofi_with_pca(data):
    """
    Integrate multi-level OFI metrics using PCA.
    """
    # Extract OFI columns
    ofi_columns = [f'ofi_{level:02d}' for level in range(5)]
    ofi_data = data[ofi_columns]
    
    # Check if there are at least 2 rows of valid data
    if len(ofi_data.dropna()) >= 2:
        # Apply PCA
        pca = PCA(n_components=1)
        ofi_pca_transformed = pca.fit_transform(ofi_data.dropna())
        
        # Create a new column for PCA-transformed OFI
        data['ofi_pca'] = np.nan  # Initialize with NaN
        valid_indices = ofi_data.dropna().index  # Indices of rows without missing values
        data.loc[valid_indices, 'ofi_pca'] = ofi_pca_transformed.flatten()
    else:
        # If not enough data, set ofi_pca to NaN for all rows
        data['ofi_pca'] = np.nan
    
    return data

def save_processed_data(data, output_folder):
    """
    Save the processed data to the output folder.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the processed data
    output_file = os.path.join(output_folder, 'processed_data.csv')
    data.to_csv(output_file, index=False, na_rep='NaN')  # Use na_rep='NaN'
    print(f"Processed data saved to {output_file}")

def process_single_file(input_file, output_folder):
    """
    Process a single cleaned CSV file and save the results.
    """
    # Load the cleaned data
    data = pd.read_csv(input_file, na_values=[''])
    
    # Calculate OFI
    data = calculate_ofi(data)
    
    # Integrate OFI with PCA
    data = integrate_ofi_with_pca(data)
    
    # Save the processed data
    file_name = os.path.basename(input_file).replace('cleaned_', 'processed_')
    output_file = os.path.join(output_folder, file_name)
    data.to_csv(output_file, index=False, na_rep='NaN')  # Use na_rep='NaN'
    print(f"Processed data saved to {output_file}")

def process_files(input_folder, output_folder):
    """
    Process all cleaned CSV files in the input folder.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # List all cleaned CSV files
    cleaned_files = [f for f in os.listdir(input_folder) if f.startswith('cleaned_')]
    
    # Process each file
    for file in cleaned_files:
        input_file = os.path.join(input_folder, file)
        process_single_file(input_file, output_folder)

if __name__ == "__main__":
    # Define input and output folders
    input_folder = 'data/cleaned'
    output_folder = 'data/processed'
    
    # Process all files
    process_files(input_folder, output_folder)