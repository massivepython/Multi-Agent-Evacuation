"""
Batch runner for testing different altruism levels.
Runs multiple simulations with varying altruism parameters and collects results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mesa.batchrunner import batch_run
from model import EvacuationModel


def run_altruism_experiment():
    """
    Run batch experiments to test different altruism levels.
    
    Tests different mean altruism levels (from selfish to altruistic)
    and compares evacuation outcomes.
    """
    
    print("Starting batch experiment to test altruism levels...")
    print("This may take several minutes...\n")
    
    # Define parameters to test
    fixed_params = {
        "width": 50,
        "height": 30,
        "num_people": 50,
        "num_obstacles": 10,
        "smoke_level": 50.0,  # Constant smoke level
        "health_min": 50,
        "health_max": 100
    }
    
    # Vary altruism levels from very selfish to very altruistic
    variable_params = {
        "altruism_mean": np.arange(0.0, 1.1, 0.1),  # 0.0, 0.1, ..., 1.0
        "altruism_std": [0.2]  # Keep standard deviation constant
    }
    
    # Number of iterations per parameter combination
    iterations = 10
    max_steps = 500
    
    # Run batch
    results = batch_run(
        EvacuationModel,
        parameters={**fixed_params, **variable_params},
        iterations=iterations,
        max_steps=max_steps,
        number_processes=1,
        data_collection_period=1,
        display_progress=True
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save raw results
    df.to_csv("batch_results.csv", index=False)
    print("\nRaw results saved to 'batch_results.csv'")
    
    # Analyze results
    analyze_results(df)
    
    return df


def analyze_results(df):
    """
    Analyze and visualize the batch run results.
    
    Creates plots showing:
    1. Evacuation success rate vs altruism
    2. Average survival time vs altruism
    3. Death rate vs altruism
    """
    
    print("\n" + "="*60)
    print("ANALYSIS: Impact of Altruism on Evacuation Outcomes")
    print("="*60 + "\n")
    
    # Get final step for each run
    final_df = df.loc[df.groupby(['RunId'])['Step'].idxmax()]
    
    # Group by altruism level
    grouped = final_df.groupby('altruism_mean').agg({
        'Escaped': ['mean', 'std'],
        'Dead': ['mean', 'std'],
        'Remaining': ['mean', 'std'],
    }).reset_index()
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    
    # Print summary statistics
    print("Summary by Altruism Level:")
    print("-" * 60)
    for _, row in grouped.iterrows():
        altruism = row['altruism_mean']
        escaped = row['Escaped_mean']
        dead = row['Dead_mean']
        print(f"Altruism {altruism:.1f}: "
              f"Escaped={escaped:.1f}±{row['Escaped_std']:.1f}, "
              f"Dead={dead:.1f}±{row['Dead_std']:.1f}")
    
    # Find optimal altruism level
    best_altruism = grouped.loc[grouped['Escaped_mean'].idxmax(), 'altruism_mean']
    best_escaped = grouped.loc[grouped['Escaped_mean'].idxmax(), 'Escaped_mean']
    
    print("\n" + "="*60)
    print(f"OPTIMAL ALTRUISM LEVEL: {best_altruism:.1f}")
    print(f"Average Escaped: {best_escaped:.1f} out of 50 people")
    print("="*60 + "\n")
    
    # Create visualizations
    create_plots(grouped, best_altruism)


def create_plots(grouped, best_altruism):
    """Create visualization plots of the results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Impact of Altruism on Emergency Evacuation Outcomes', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Escaped count
    ax1 = axes[0, 0]
    ax1.errorbar(grouped['altruism_mean'], grouped['Escaped_mean'], 
                 yerr=grouped['Escaped_std'], marker='o', capsize=5, 
                 linewidth=2, markersize=8, color='green')
    ax1.axvline(best_altruism, color='red', linestyle='--', 
                label=f'Optimal ({best_altruism:.1f})')
    ax1.set_xlabel('Mean Altruism Level', fontsize=12)
    ax1.set_ylabel('Number of People Escaped', fontsize=12)
    ax1.set_title('Evacuation Success Rate', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Death count
    ax2 = axes[0, 1]
    ax2.errorbar(grouped['altruism_mean'], grouped['Dead_mean'], 
                 yerr=grouped['Dead_std'], marker='o', capsize=5, 
                 linewidth=2, markersize=8, color='red')
    ax2.axvline(best_altruism, color='green', linestyle='--',
                label=f'Optimal ({best_altruism:.1f})')
    ax2.set_xlabel('Mean Altruism Level', fontsize=12)
    ax2.set_ylabel('Number of Deaths', fontsize=12)
    ax2.set_title('Death Rate', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Survival rate percentage
    ax3 = axes[1, 0]
    survival_rate = (grouped['Escaped_mean'] / 50) * 100
    ax3.plot(grouped['altruism_mean'], survival_rate, marker='o', 
             linewidth=2, markersize=8, color='blue')
    ax3.axvline(best_altruism, color='red', linestyle='--',
                label=f'Optimal ({best_altruism:.1f})')
    ax3.set_xlabel('Mean Altruism Level', fontsize=12)
    ax3.set_ylabel('Survival Rate (%)', fontsize=12)
    ax3.set_title('Survival Rate Percentage', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([0, 100])
    
    # Plot 4: Comparison bar chart
    ax4 = axes[1, 1]
    x_pos = np.arange(len(grouped['altruism_mean']))
    width = 0.35
    ax4.bar(x_pos - width/2, grouped['Escaped_mean'], width, 
            label='Escaped', color='green', alpha=0.7)
    ax4.bar(x_pos + width/2, grouped['Dead_mean'], width, 
            label='Dead', color='red', alpha=0.7)
    ax4.set_xlabel('Mean Altruism Level', fontsize=12)
    ax4.set_ylabel('Number of People', fontsize=12)
    ax4.set_title('Escaped vs Dead Comparison', fontsize=13, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'{x:.1f}' for x in grouped['altruism_mean']], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('altruism_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to 'altruism_analysis.png'")
    print("\nTip: You can also view the plot by uncommenting plt.show()")
    # plt.show()  # Uncomment to display the plot


def quick_test():
    """
    Run a quick test with just 3 altruism levels for faster results.
    Useful for testing before running the full experiment.
    """
    print("Running quick test with 3 altruism levels...")
    
    fixed_params = {
        "width": 50,
        "height": 30,
        "num_people": 50,
        "num_obstacles": 10,
        "smoke_level": 50.0,
        "health_min": 50,
        "health_max": 100,
        "altruism_std": 0.2
    }
    
    variable_params = {
        "altruism_mean": [0.0, 0.5, 1.0]  # Selfish, Neutral, Altruistic
    }
    
    results = batch_run(
        EvacuationModel,
        parameters={**fixed_params, **variable_params},
        iterations=5,
        max_steps=300,
        number_processes=1,
        data_collection_period=1,
        display_progress=True
    )
    
    df = pd.DataFrame(results)
    final_df = df.loc[df.groupby(['RunId'])['Step'].idxmax()]
    
    summary = final_df.groupby('altruism_mean').agg({
        'Escaped': 'mean',
        'Dead': 'mean'
    })
    
    print("\nQuick Test Results:")
    print(summary)
    print("\nRun full experiment with run_altruism_experiment() for complete analysis.")
    
    return df


if __name__ == "__main__":
    # Run full experiment
    # Uncomment the line below to run full experiment:
    df = run_altruism_experiment()
    
    # Or run quick test:
    # df = quick_test()

