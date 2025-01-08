import pandas as pd
import os

# Define paths relative to the script's location
script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
root_dir = os.path.abspath(os.path.join(script_dir, ".."))  # Move up one level to the root folder

input_folder = os.path.join(root_dir, "data", "processed")  # Input folder for CSV files
output_folder = os.path.join(root_dir, "data", "parquet")   # Output folder for Parquet files

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all CSV files in the input folder
for file in os.listdir(input_folder):
    if file.endswith('.csv'):
        print(f"Converting {file} to Parquet...")
        
        # Read the CSV file
        csv_path = os.path.join(input_folder, file)
        df = pd.read_csv(csv_path)
        
        # Write the DataFrame to a Parquet file
        parquet_file = file.replace('.csv', '.parquet')
        parquet_path = os.path.join(output_folder, parquet_file)
        df.to_parquet(parquet_path)

        print(f"Saved {parquet_file} to {output_folder}")
