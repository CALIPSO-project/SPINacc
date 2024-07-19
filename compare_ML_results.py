import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def relative_difference(x, y):
    """Calculate the relative difference between two values."""
    return abs(x - y) / ((x + y) / 2) * 100

def process_and_plot(file_10y, file_1y):
    # Load the datasets
    df_10y = pd.read_csv(file_10y)
    df_1y = pd.read_csv(file_1y)

    # Join the dataframes on identifying columns
    df_merged = pd.merge(df_10y, df_1y, on=["comp", "var", "dim_1", "ind_1"], suffixes=('_10y', '_1y'))

    # Numeric columns to compare
    numeric_columns = ['R2', 'RMSE', 'slope', 'reMSE', 'dNRMSE', 'sNRMSE', 'iNRMSE', 'f_SB', 'f_SDSD', 'f_LSC']
    results_df = pd.DataFrame()

    # Compute the relative differences for each cell and save to results_df
    for column in numeric_columns:
        results_df[column] = relative_difference(df_merged[column + '_10y'], df_merged[column + '_1y'])

    # Save the detailed results to a CSV file
    results_df.to_csv('detailed_relative_differences.csv', index=False)

    # Plotting average relative differences
    fig, ax = plt.subplots(figsize=(14, 8))
    averages = results_df.mean()
    colors = ['green' if x < 10 else 'orange' if x < 30 else 'red' for x in averages]
    ax.bar(numeric_columns, averages, color=colors)
    
    ax.set_xticks(np.arange(len(numeric_columns)))
    ax.set_xticklabels(numeric_columns, rotation=45)
    ax.set_ylabel('Average Relative Difference (%)')
    ax.set_title('Average Relative Differences Across Metrics')
    plt.tight_layout()

    # Save the figure
    plt.savefig('relative_differences.png')
    plt.show()

# Example usage
process_and_plot('MLacc_1Y_forcing.csv', 'MLacc_10Y_forcing.csv')

