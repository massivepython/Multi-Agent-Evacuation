"""
Emergency Evacuation Multi-Agent System
Main entry point for the simulation.

This file provides a simple interface to run the model.
"""

from model import EvacuationModel
import sys


def main():
    """Main entry point."""
    print("="*70)
    print("Emergency Evacuation Multi-Agent System")
    print("="*70)
    print()
    print("Options:")
    print("1. Run interactive visualization (recommended)")
    print("2. Run batch experiments")
    print("3. Run single simulation")
    print()
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        print("\nLaunching visualization server...")
        print("Open your browser to http://localhost:8521")
        from visualization import server
        server.launch()
        
    elif choice == "2":
        print("\nRunning batch experiments...")
        print("This will test different altruism levels and create analysis plots.")
        confirm = input("This may take several minutes. Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            from batch_run import run_altruism_experiment
            run_altruism_experiment()
        else:
            print("Cancelled.")
            
    elif choice == "3":
        print("\nRunning single simulation...")
        
        # Get parameters
        try:
            num_people = int(input("Number of people (default 50): ") or "50")
            altruism = float(input("Mean altruism level 0-1 (default 0.5): ") or "0.5")
        except ValueError:
            print("Invalid input. Using defaults.")
            num_people = 50
            altruism = 0.5
        
        # Create and run model
        model = EvacuationModel(
            num_people=num_people,
            altruism_mean=altruism
        )
        
        print("\nRunning simulation...")
        model.run_model(max_steps=500)
        
        # Print results
        print("\n" + "="*70)
        print("SIMULATION RESULTS")
        print("="*70)
        print(f"Total people: {num_people}")
        print(f"Escaped: {model.escaped_count}")
        print(f"Dead: {model.dead_count}")
        print(f"Remaining: {num_people - model.escaped_count - model.dead_count}")
        print(f"Survival rate: {(model.escaped_count / num_people) * 100:.1f}%")
        print(f"Steps taken: {model.steps}")
        print("="*70)
        
    else:
        print("Invalid choice. Exiting.")
        

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
        sys.exit(0)

