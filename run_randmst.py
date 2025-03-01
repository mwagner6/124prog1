import subprocess
import pandas as pd
import numpy as np

def run_randmst(n, num_trials, dimension):
    """Run randmst program and return average weight"""
    try:
        # Run the command and capture output
        result = subprocess.run(
            ['./randmst', '0', str(n), str(num_trials), str(dimension)],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the first line which contains: average numpoints numtrials dimension
        output = result.stdout.split('\n')[0]
        avg_weight = float(output.split()[0])
        return avg_weight
    
    except subprocess.CalledProcessError as e:
        print(f"Error running randmst: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def main():
    # Parameters
    num_trials = 5  # Run at least 5 times for each n
    dimensions = [0, 1, 2, 3, 4]  # All dimensions to test
    
    # Different n values for complete graphs (dim 0, 2, 3, 4) and hypercube (dim 1)
    complete_n = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    hypercube_n = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
    
    results = []
    
    for dim in dimensions:
        print(f"\nTesting dimension {dim}")
        n_values = hypercube_n if dim == 1 else complete_n
        
        for n in n_values:
            print(f"Testing n={n}")
            avg_weight = run_randmst(n, num_trials, dim)
            if avg_weight is not None:
                results.append({
                    'dimension': dim,
                    'n': n,
                    'avg_weight': avg_weight
                })
    
    # Convert results to DataFrame and display
    df = pd.DataFrame(results)
    
    # Display results grouped by dimension
    for dim in dimensions:
        dim_results = df[df['dimension'] == dim]
        print(f"\nResults for dimension {dim}:")
        print(dim_results.to_string(index=False))
        
        # Save results to CSV
        dim_results.to_csv(f'randmst_results_dim_{dim}.csv', index=False)

if __name__ == "__main__":
    main()