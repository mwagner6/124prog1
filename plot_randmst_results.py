import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import multiprocessing as mp
from functools import partial

# Define fitting functions
def power_law(x, a, b):
    return a * np.power(x, b)

def log_law(x, a, b):
    return a * np.log(x) + b

def sqrt_law(x, a):
    return a * np.sqrt(x)

def linear_law(x, a, b):
    return a * x + b

# Calculate RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Function to process all dimensions in a single plot
def process_all_dimensions(results, markers, colors):
    # Create a figure with 5 subplots arranged vertically
    fig, axes = plt.subplots(5, 1, figsize=(15, 25))
    
    for dim in results:
        if dim not in results:
            continue
            
        ax = axes[dim]  # Get the subplot for this dimension
        df = results[dim]
        n_values = df['n'].values
        weights = df['avg_weight'].values
        
        try:
            # Power law fit
            popt_power, _ = curve_fit(power_law, n_values, weights)
            power_fit = power_law(n_values, *popt_power)
            
            # Log law fit
            popt_log, _ = curve_fit(log_law, n_values, weights)
            log_fit = log_law(n_values, *popt_log)
            
            # Sqrt law fit
            popt_sqrt, _ = curve_fit(sqrt_law, n_values, weights)
            sqrt_fit = sqrt_law(n_values, *popt_sqrt)
            
            # Linear law fit
            popt_linear, _ = curve_fit(linear_law, n_values, weights)
            linear_fit = linear_law(n_values, *popt_linear)
            
            # Calculate RMSE for each fit
            rmse_power = calculate_rmse(weights, power_fit)
            rmse_log = calculate_rmse(weights, log_fit)
            rmse_sqrt = calculate_rmse(weights, sqrt_fit)
            rmse_linear = calculate_rmse(weights, linear_fit)
            
            print(f"\nDimension {dim} fitting results:")
            print(f"Power law: f(n) = {popt_power[0]:.4f} * n^{popt_power[1]:.4f} (RMSE = {rmse_power:.4f})")
            print(f"Log law: f(n) = {popt_log[0]:.4f} * ln(n) + {popt_log[1]:.4f} (RMSE = {rmse_log:.4f})")
            print(f"Square root: f(n) = {popt_sqrt[0]:.4f} * sqrt(n) (RMSE = {rmse_sqrt:.4f})")
            print(f"Linear: f(n) = {popt_linear[0]:.4f} * n + {popt_linear[1]:.4f} (RMSE = {rmse_linear:.4f})")
            
            # Plot data points and fits on the subplot
            ax.plot(n_values, weights, 
                   marker=markers[dim],
                   color=colors[dim],
                   label='Data points',
                   linestyle='',
                   markersize=8)
            
            # Plot fits with different line styles
            ax.plot(n_values, power_fit, color=colors[0], linestyle='-',
                   label=f'Power: {popt_power[0]:.2f}n^{popt_power[1]:.2f} (RMSE={rmse_power:.2e})')
            ax.plot(n_values, log_fit, color=colors[1], linestyle='--',
                   label=f'Log: {popt_log[0]:.2f}ln(n) + {popt_log[1]:.2f} (RMSE={rmse_log:.2e})')
            ax.plot(n_values, sqrt_fit, color=colors[2], linestyle=':',
                   label=f'Sqrt: {popt_sqrt[0]:.2f}√n (RMSE={rmse_sqrt:.2e})')
            ax.plot(n_values, linear_fit, color=colors[3], linestyle='-.',
                   label=f'Linear: {popt_linear[0]:.2f}n + {popt_linear[1]:.2f} (RMSE={rmse_linear:.2e})')
            
            # Customize each subplot
            ax.grid(True, which="both", ls="-", alpha=0.2)
            ax.set_title(f'Dimension {dim}', fontsize=12)
            ax.set_xlabel('Number of Vertices (n)', fontsize=10)
            ax.set_ylabel('Average MST Weight', fontsize=10)
            ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.0, 0.5))
            
        except RuntimeError:
            print(f"Could not fit functions for dimension {dim}")
            ax.text(0.5, 0.5, f'Could not fit functions for dimension {dim}',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
    
    # Add an overall title to the figure
    fig.suptitle('Minimum Spanning Tree Average Weights by Dimension', fontsize=16, y=0.95)
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
    # Save the plot with extra space for the legends
    plt.savefig('combined.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Read all dimension CSV files
    dimensions = range(5)  # 0 through 4
    results = {}

    for dim in dimensions:
        filename = f'randmst_results_dim_{dim}.csv'
        try:
            df = pd.read_csv(filename)
            results[dim] = df
        except FileNotFoundError:
            print(f"Warning: Could not find {filename}")

    # Colors and markers for consistency across plots
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e']  # Distinct colors

    # Process all dimensions in a single plot
    process_all_dimensions(results, markers, colors)

if __name__ == "__main__":
    main()
