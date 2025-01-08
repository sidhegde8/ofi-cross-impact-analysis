import os
import pandas as pd
import matplotlib.pyplot as plt

# Set up directories
output_folder = 'results'  # Update this path to your output folder
best_lagged_results_folder = os.path.join(output_folder, 'best_lagged_results')
feature_importance_folder = os.path.join(output_folder, 'feature_importance')
feature_importances_plots_folder = os.path.join(feature_importance_folder, 'plots')
feature_importances_tables_folder = os.path.join(feature_importance_folder, 'tables')

# Create subfolders for plots and tables if they don't exist
os.makedirs(feature_importances_plots_folder, exist_ok=True)
os.makedirs(feature_importances_tables_folder, exist_ok=True)

def load_feature_importances(file_path):
    """
    Load feature importances from a text file in the best_lagged_results folder.
    """
    feature_importances = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Flag to indicate when we've reached the feature importances section
        in_feature_section = False
        for line in lines:
            # Skip the header and R-squared lines
            if line.startswith("Random Forest Model Summary:") or line.startswith("R-squared:"):
                continue
            # Start processing feature importances after the "Feature Importances:" line
            if line.strip() == "Feature Importances:":
                in_feature_section = True
                continue
            # Only process lines in the feature importances section
            if in_feature_section and ':' in line:
                try:
                    feature, importance = line.strip().split(':')
                    importance_value = float(importance.strip())
                    feature_importances[feature.strip()] = importance_value
                except ValueError:
                    print(f"Warning: Skipping malformed line in {file_path}: {line.strip()}")
    return feature_importances

def visualize_feature_importances(feature_importances, filename):
    """
    Generate a bar plot of feature importances and save it as an image.
    """
    features = list(feature_importances.keys())
    importances = list(feature_importances.values())

    plt.figure(figsize=(10, 6))
    plt.bar(features, importances, color='skyblue')
    plt.title(f"Feature Importances: {filename.replace('_', ' ')}")
    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plot_filename = os.path.join(feature_importances_plots_folder, f"{filename}.png")
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()

def save_feature_importances_table(feature_importances, filename):
    """
    Save feature importances as a table in a CSV file.
    """
    # Convert feature importances to a DataFrame
    df = pd.DataFrame(list(feature_importances.items()), columns=['Feature', 'Importance'])

    # Save the table as a CSV file
    table_filename = os.path.join(feature_importances_tables_folder, f"{filename}.csv")
    df.to_csv(table_filename, index=False)

def main():
    # Loop through all feature importance files in the best_lagged_results folder
    for filename in os.listdir(best_lagged_results_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(best_lagged_results_folder, filename)
            print(f"Processing file: {filename}")

            # Load feature importances
            feature_importances = load_feature_importances(file_path)

            # Generate and save visualization
            visualize_feature_importances(feature_importances, filename.replace('.txt', ''))

            # Save feature importances as a table
            save_feature_importances_table(feature_importances, filename.replace('.txt', ''))

    print("Feature importance visualizations saved to:", feature_importances_plots_folder)
    print("Feature importance tables saved to:", feature_importances_tables_folder)

if __name__ == "__main__":
    main()