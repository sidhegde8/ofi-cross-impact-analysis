import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up directories
output_folder = 'results'  # Path to the main results folder
sector_analysis_folder = os.path.join(output_folder, 'sector_analysis')  # New folder for sector-level analysis

# Create the sector_analysis folder if it doesn't exist
os.makedirs(sector_analysis_folder, exist_ok=True)

# Sector mapping
sector_mapping = {
    'AAPL': 'Tech',
    'TSLA': 'Consumer Discretionary',
    'AMGN': 'Healthcare',
    'JPM': 'Financials',
    'XOM': 'Energy'
}

# Function to extract R-squared values from the output files
def extract_r_squared_from_file(file_path):
    """
    Extract the R-squared value from a results file.
    """
    with open(file_path, 'r') as f:
        content = f.read()
        match = re.search(r"R-squared: ([\d.]+)", content)
        if match:
            return float(match.group(1))
    return None

# Function to analyze sector-level cross-impact
def analyze_sector_cross_impact_from_results(output_folder, sector_mapping):
    sector_results = {}  # Dictionary to store results for each sector pair
    
    # List all output files
    result_files = [f for f in os.listdir(output_folder) if f.startswith('best_lagged_') and f.endswith('.txt')]
    
    for file in result_files:
        # Extract stock pair and lag from the filename
        match = re.match(r"best_lagged_(\w+)_to_(\w+)_(.+)\.txt", file)
        if match:
            stock_a, stock_b, lag = match.groups()
            
            # Determine the sectors for the stock pair
            sector_a = sector_mapping.get(stock_a, 'Unknown')
            sector_b = sector_mapping.get(stock_b, 'Unknown')
            
            # Create a key for the sector pair (e.g., "Tech → Healthcare")
            sector_pair = (sector_a, sector_b)
            
            # Initialize the sector pair in the results dictionary if it doesn't exist
            if sector_pair not in sector_results:
                sector_results[sector_pair] = []
            
            # Extract R-squared value from the file
            file_path = os.path.join(output_folder, file)
            r_squared = extract_r_squared_from_file(file_path)
            
            if r_squared is not None:
                sector_results[sector_pair].append(r_squared)
    
    return sector_results

# Main function
def main():
    # Analyze sector-level cross-impact using precomputed results
    sector_results = analyze_sector_cross_impact_from_results(output_folder, sector_mapping)

    # Calculate average R-squared for each sector pair
    sector_pair_avg_r_squared = {pair: np.mean(values) for pair, values in sector_results.items()}

    # Convert results to a DataFrame for easier visualization
    sectors = sorted(set(sector_mapping.values()))  # Unique sectors
    heatmap_data = pd.DataFrame(index=sectors, columns=sectors, dtype=float)

    # Populate the heatmap data
    for (sector_a, sector_b), avg_r_squared in sector_pair_avg_r_squared.items():
        heatmap_data.loc[sector_a, sector_b] = avg_r_squared

    # Fill NaN values with 0 (or any other placeholder)
    heatmap_data = heatmap_data.fillna(0)

    # Save the heatmap data to a CSV file
    csv_path = os.path.join(sector_analysis_folder, 'sector_pair_heatmap_data.csv')
    heatmap_data.to_csv(csv_path)
    print(f"Heatmap data saved to: {csv_path}")

    # Visualize results as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data,
        annot=True,  # Display values in each cell
        fmt=".3f",   # Format values to 3 decimal places
        cmap="YlGnBu",  # Color map
        linewidths=0.5,  # Add grid lines
        cbar_kws={'label': 'Average R-squared'}  # Add color bar label
    )
    plt.title('Cross-Impact Heatmap by Sector Pair')
    plt.xlabel('Target Sector')
    plt.ylabel('Source Sector')
    
    # Save the heatmap to the new sector_analysis folder
    plot_path = os.path.join(sector_analysis_folder, 'sector_pair_cross_impact_heatmap.png')
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.savefig(plot_path)
    plt.close()

    print(f"Sector-level cross-impact analysis complete. Results saved to: {sector_analysis_folder}")

if __name__ == "__main__":
    main()