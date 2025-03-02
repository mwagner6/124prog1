import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def process_timing_data(results, markers, colors):
    # Create a figure with 3 subplots stacked vertically
    fig, (ax_total, ax_gen, ax_mst) = plt.subplots(3, 1, figsize=(15, 20))
    
    # Plot for each dimension
    for dim in results:
        df = results[dim]
        n_values = df['n'].values
        total_times = df['total_time_ms'].values
        gen_times = df['gen_time_ms'].values
        mst_times = df['mst_time_ms'].values
        
        # Plot total time
        ax_total.plot(n_values, total_times,
                     marker=markers[dim],
                     color=colors[dim],
                     label=f'Dimension {dim}',
                     linestyle='-',
                     markersize=8)
        
        # Plot generation time
        ax_gen.plot(n_values, gen_times,
                   marker=markers[dim],
                   color=colors[dim],
                   label=f'Dimension {dim}',
                   linestyle='-',
                   markersize=8)
        
        # Plot MST time
        ax_mst.plot(n_values, mst_times,
                   marker=markers[dim],
                   color=colors[dim],
                   label=f'Dimension {dim}',
                   linestyle='-',
                   markersize=8)
    
    # Customize total time subplot
    ax_total.set_title('Total Runtime', fontsize=14)
    ax_total.set_ylabel('Time (ms)', fontsize=12)
    ax_total.grid(True, alpha=0.2)
    ax_total.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax_total.set_xscale('log')
    ax_total.set_yscale('log')
    
    # Customize generation time subplot
    ax_gen.set_title('Graph Generation Time', fontsize=14)
    ax_gen.set_ylabel('Time (ms)', fontsize=12)
    ax_gen.grid(True, alpha=0.2)
    ax_gen.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax_gen.set_xscale('log')
    ax_gen.set_yscale('log')
    
    # Customize MST time subplot
    ax_mst.set_title('MST Computation Time', fontsize=14)
    ax_mst.set_xlabel('Number of Vertices (n)', fontsize=12)
    ax_mst.set_ylabel('Time (ms)', fontsize=12)
    ax_mst.grid(True, alpha=0.2)
    ax_mst.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax_mst.set_xscale('log')
    ax_mst.set_yscale('log')
    
    # Add an overall title
    fig.suptitle('Runtime Analysis by Dimension', fontsize=16, y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('combined_times.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Read all dimension CSV files
    dimensions = range(5)  # 0 through 4
    results = {}
    
    for dim in dimensions:
        filename = f'randmst_results_weightandtime_dim_{dim}.csv'
        try:
            df = pd.read_csv(filename)
            results[dim] = df
        except FileNotFoundError:
            print(f"Warning: Could not find {filename}")
    
    # Colors and markers for consistency across plots
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e']
    
    # Process timing data
    process_timing_data(results, markers, colors)

if __name__ == "__main__":
    main() 