import pandas as pd
import os

input_folder = '/path/to/csv/files'
output_folder = '/path/to/parquet/files'
os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith('.csv'):
        print(f"Converting {file} to Parquet...")
        df = pd.read_csv(os.path.join(input_folder, file))
        df.to_parquet(os.path.join(output_folder, file.replace('.csv', '.parquet')))
