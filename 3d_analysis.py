"""
3D Analysis Script for Emergency Evacuation Simulation

This script performs a grid search over two parameters:
1. Helping Speed Impact (X-axis)
2. Altruistic Population Percentage (Y-axis)

Output:
- 3D Surface Plot of Average Deaths
- Heatmap of Average Deaths
- CSV of raw results

Modes:
- LIGHT: Coarse grid (step 10), fewer reps (10). Approx 1,000 simulations. Good for testing.
- HEAVY: Fine grid (step 5), many reps (30). Approx 12,000 simulations. High precision.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing
import time
from itertools import product
import sys

# Import the model
from model import EvacuationModel

# Configuration Constants
LIGHT_MODE = {
    'speed_range': range(10, 105, 10),     # 10, 20, ..., 100
    'altruism_range': range(10, 105, 10),  # 10, 20, ..., 100
    'replications': 10
}

HEAVY_MODE = {
    'speed_range': range(5, 105, 5),       # 5, 10, ..., 100
    'altruism_range': range(5, 105, 5),    # 5, 10, ..., 100
    'replications': 30
}

ULTRA_MODE = {
    'speed_range': range(2, 102, 2),       # 2, 4, ..., 100
    'altruism_range': range(2, 102, 2),    # 2, 4, ..., 100
    'replications': 100
}

ROBUST_MODE = {
    'speed_range': range(0, 105, 5),       # 0, 5, ..., 100
    'altruism_range': range(0, 105, 5),    # 0, 5, ..., 100
    'replications': 70
}

PRECISE_MODE = {
    'speed_range': range(0, 104, 4),       # 0, 4, 8, ..., 100
    'altruism_range': range(0, 104, 4),    # 0, 4, 8, ..., 100
    'replications': 100
}

def run_single_simulation(params):
    """
    Worker function for multiprocessing.
    Runs a single instance of the model with specific parameters.
    """
    speed_pct, altruism_pct, seed = params
    
    # Initialize model with specific parameters
    # We use a fixed seed if provided to ensure reproducibility across runs if needed,
    # though typically we want random variation per replication.
    # For this script, we rely on numpy's internal randomness which is generally 
    # process-safe enough for this level of simulation, but explicitly re-seeding 
    # based on process ID or unique index is safer in some contexts.
    # Mesa handles its own random number generator.
    
    model = EvacuationModel(
        helping_speed_percentage=speed_pct,
        altruistic_population_percent=altruism_pct,
        run_mode="Interactive Simulation", # Headless mode effectively
        # Standard fixed parameters
        width=50, height=30, num_people=150, smoke_level=50.0
    )
    
    # Run until completion or max steps
    model.run_model(max_steps=1000)
    
    return params, model.dead_count

def generate_plot(df, mode_name):
    """Generates and saves the 3D plot and Heatmap."""
    
    # Pivot data for plotting
    # X = Speed, Y = Altruism, Z = Dead Count
    pivot_table = df.pivot_table(
        values='dead_count', 
        index='altruism_pct',
        columns='speed_pct',
        aggfunc='mean'
    )
    
    X = pivot_table.columns.values
    Y = pivot_table.index.values
    X_grid, Y_grid = np.meshgrid(X, Y)
    Z_grid = pivot_table.values

    # --- 3D Surface Plot ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(
        X_grid, Y_grid, Z_grid, 
        cmap='viridis_r', # Reverse viridis: Blue=Low Deaths (Good), Yellow=High Deaths (Bad)
        edgecolor='none',
        alpha=0.9
    )
    
    ax.set_xlabel('Helping Speed Impact (%)')
    ax.set_ylabel('Altruism Population (%)')
    ax.set_zlabel('Average Deaths')
    ax.set_title(f'Impact of Altruism and Helping Speed on Survival ({mode_name})')
    
    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Average Deaths')
    
    # Save 3D Plot
    plt.savefig('analysis_3d_surface.png', dpi=300)
    print("Saved 3D plot to 'analysis_3d_surface.png'")
    
    # --- Heatmap ---
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(X_grid, Y_grid, Z_grid, cmap='viridis_r', shading='auto')
    plt.colorbar(label='Average Deaths')
    plt.xlabel('Helping Speed Impact (%)')
    plt.ylabel('Altruism Population (%)')
    plt.title(f'Heatmap: Average Deaths ({mode_name})')
    
    # Save Heatmap
    plt.savefig('analysis_heatmap.png', dpi=300)
    print("Saved Heatmap to 'analysis_heatmap.png'")

def main():
    print("--- 3D Analysis: Helping Speed vs Altruism ---")
    print("1. LIGHT Mode (~1,000 sims, Fast)")
    print("2. HEAVY Mode (~12,000 sims, Slow)")
    print("3. ULTRA Mode (250,000 sims, Very Slow)")
    print("4. ROBUST Mode (High Reps, 0-100 Step 5)")
    print("5. PRECISE Mode (High Reps, 0-100 Step 4)")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        print(f"Mode selected via argument: {choice}")
    else:
        choice = input("Select mode (1, 2, 3, 4, or 5): ").strip()
    
    if choice == '5':
        config = PRECISE_MODE
        mode_name = "PRECISE"
    elif choice == '4':
        config = ROBUST_MODE
        mode_name = "ROBUST"
    elif choice == '3':
        config = ULTRA_MODE
        mode_name = "ULTRA"
    elif choice == '2':
        config = HEAVY_MODE
        mode_name = "HEAVY"
    else:
        config = LIGHT_MODE
        mode_name = "LIGHT"
        
    speeds = list(config['speed_range'])
    altruisms = list(config['altruism_range'])
    reps = config['replications']
    
    total_sims = len(speeds) * len(altruisms) * reps
    print(f"\nInitializing {mode_name} mode.")
    print(f"Grid: {len(speeds)}x{len(altruisms)}")
    print(f"Replications per point: {reps}")
    print(f"Total Simulations: {total_sims}")
    print(f"Using {multiprocessing.cpu_count()} CPU cores.")
    
    # Prepare parameter list for multiprocessing
    # Structure: (speed, altruism, replication_id)
    tasks = []
    for s in speeds:
        for a in altruisms:
            for r in range(reps):
                tasks.append((s, a, r))
                
    start_time = time.time()
    
    # Run simulations in parallel
    results_list = []
    current_sim_count = 0
    start_time_sims = time.time()
    
    print("\nStarting simulations...")
    with multiprocessing.Pool() as pool:
        # Use imap_unordered to get results as they complete, for progress tracking
        for i, (params, dead_count) in enumerate(pool.imap_unordered(run_single_simulation, tasks)):
            results_list.append((params, dead_count))
            current_sim_count += 1
            
            # Print progress every 1% or significant milestone
            if current_sim_count % (total_sims // 100 or 1) == 0 or current_sim_count == total_sims:
                elapsed_time = time.time() - start_time_sims
                progress_pct = (current_sim_count / total_sims) * 100
                # Estimate remaining time
                if current_sim_count > 0:
                    time_per_sim = elapsed_time / current_sim_count
                    remaining_sims = total_sims - current_sim_count
                    estimated_remaining_time = remaining_sims * time_per_sim
                    # Clear the line and print new progress
                    sys.stdout.write(f"\rProgress: {progress_pct:.1f}% | Completed: {current_sim_count}/{total_sims} | Elapsed: {elapsed_time:.0f}s | Est. Remaining: {estimated_remaining_time:.0f}s")
                else:
                    sys.stdout.write(f"\rProgress: {progress_pct:.1f}% | Completed: {current_sim_count}/{total_sims} | Elapsed: {elapsed_time:.0f}s")
                sys.stdout.flush() # Ensure the output is immediately displayed
                
    # After the loop, print a final newline to ensure the next print statements appear on a new line.
    sys.stdout.write("\n") # Newline after the progress bar
    sys.stdout.flush()
    
    duration = time.time() - start_time
    print(f"Simulations complete in {duration:.2f} seconds ({duration/total_sims:.3f}s per sim).")
    
    # Process Results
    # Combine parameters with results
    data = []
    for params, dead_count in results_list:
        speed, altruism, _ = params
        data.append({
            'speed_pct': speed,
            'altruism_pct': altruism,
            'dead_count': dead_count
        })
        
    df = pd.DataFrame(data)
    
    # Save Raw Data
    df.to_csv('analysis_results.csv', index=False)
    print("Saved raw data to 'analysis_results.csv'")
    
    # Generate Plots
    generate_plot(df, mode_name)
    
    print("\nAnalysis Complete.")

if __name__ == '__main__':
    # On Windows, multiprocessing requires the entry point protection
    multiprocessing.freeze_support()
    main()
