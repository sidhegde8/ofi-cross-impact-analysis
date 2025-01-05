import pandas as pd
import numpy as np
import os

def load_data(file_path):
    """
    Load the raw data from a CSV file.
    - Treat empty strings as NaN.
    """
    data = pd.read_csv(file_path, na_values=[''])
    return data

def clean_data(data):
    """
    Clean the raw data:
    - Handle missing prices and sizes.
    - Remove duplicates.
    - Filter relevant columns.
    """
    for level in range(0, 5):
        data[f'bid_sz_{level:02d}'].fillna(0, inplace=True)
        data[f'ask_sz_{level:02d}'].fillna(0, inplace=True)
    
    data = data.drop_duplicates()
    
    columns_to_keep = ['ts_recv', 'symbol']
    for level in range(0, 5):  # Levels 0 to 4 (up to 5 levels)
        columns_to_keep.extend([f'bid_px_{level:02d}', f'ask_px_{level:02d}', f'bid_sz_{level:02d}', f'ask_sz_{level:02d}'])
    
    data = data[columns_to_keep]
    return data

def process_files(input_folder, output_folder):
    """
    Process all CSV files in the input folder and save cleaned data to the output folder.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    for file in files:
        file_path = os.path.join(input_folder, file)
        data = load_data(file_path)
        
        cleaned_data = clean_data(data)
        
        output_file = os.path.join(output_folder, f"cleaned_{file}")
        cleaned_data.to_csv(output_file, index=False, na_rep='NaN')  
        print(f"Data cleaned and saved to {output_file}")

if __name__ == "__main__":
    input_folder = "data/raw" 
    output_folder = "data/cleaned"  
    
    # Process all files
    process_files(input_folder, output_folder)