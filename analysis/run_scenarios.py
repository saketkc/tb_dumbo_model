"""
TB Model Scenario Runner
=======================

This script runs different TB modeling scenarios for workshop exercises.
Participants can modify this to explore different intervention strategies.
"""

import sys
import os
sys.path.append('../src')

from tb_model import TBModel
from interventions import InterventionManager
import yaml
import matplotlib.pyplot as plt
import pandas as pd


class ScenarioRunner:
    """
    Runs multiple TB model scenarios for comparison.

    This class is designed for workshop participants to experiment with
    different parameter sets and intervention combinations.
    """

    def __init__(self):
        """Initialize scenario runner."""
        self.scenarios = {}
        self.results = {}

    def add_scenario(self, name, config_file=None, parameters=None):
        """
        Add a scenario to run.

        Args:
            name: Scenario name
            config_file: Path to YAML config file
            parameters: Dictionary of parameters to override
        """
        scenario = {
            'config_file': config_file,
            'parameters': parameters or {},
            'model': None,
            'results': None
        }

        self.scenarios[name] = scenario
        print(f"Added scenario: {name}")

    def run_scenario(self, name, years=10):
        """Run a specific scenario."""
        if name not in self.scenarios:
            print(f"Scenario '{name}' not found!")
            return

        print(f"\nRunning scenario: {name}")
        print("-" * 40)

        scenario = self.scenarios[name]

        # Initialize model
        config_file = scenario['config_file'] or '../data/parameters.yaml'
        model = TBModel(config_file)

        # Apply parameter overrides
        for param, value in scenario['parameters'].items():
            if param in model.params:
                print(f"  Overriding {param}: {model.params[param]} → {value}")
                model.params[param] = value

        # Run simulation
        results = model.run_simulation(years=years)

        # Store results
        scenario['model'] = model
        scenario['results'] = results
        self.results[name] = results

        print(f"Scenario '{name}' completed")
        return results

    def run_all_scenarios(self, years=10):
        """Run all configured scenarios."""
        print("Running all scenarios...")
        print("=" * 50)

        for name in self.scenarios.keys():
            self.run_scenario(name, years)

    def compare_scenarios(self, metrics=['prevalence', 'incidence'], save_path='../results/scenario_comparison.png'):
        """
        Create comparison plots for different scenarios.

        Args:
            metrics: List of metrics to compare
            save_path: Path to save comparison plot
        """
        if not self.results:
            print("No results to compare. Run scenarios first.")
            return

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        colors = plt.cm.tab10(range(len(self.results)))

        for i, metric in enumerate(metrics):
            for j, (scenario_name, results) in enumerate(self.results.items()):
                time = results['time']
                values = results[metric]

                axes[i].plot(time, values, label=scenario_name,
                           color=colors[j], linewidth=2)

            axes[i].set_title(f'TB {metric.title()}')
            axes[i].set_xlabel('Years')

            if metric == 'prevalence':
                axes[i].set_ylabel('Cases per 100,000')
            elif metric == 'incidence':
                axes[i].set_ylabel('New cases per 100,000 per year')
            else:
                axes[i].set_ylabel(metric)

            axes[i].grid(True, alpha=0.3)
            axes[i].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
        plt.show()

    def save_comparison_data(self, filename='../results/scenario_comparison.csv'):
        """Save comparison data to CSV file."""
        if not self.results:
            print("No results to save.")
            return

        # Combine all scenario results
        combined_data = []

        for scenario_name, results in self.results.items():
            df = pd.DataFrame(results)
            df['scenario'] = scenario_name
            combined_data.append(df)

        # Concatenate all dataframes
        final_df = pd.concat(combined_data, ignore_index=True)

        # Save to file
        final_df.to_csv(filename, index=False)
        print(f"Comparison data saved to {filename}")

    def print_scenario_summary(self):
        """Print summary of all scenario results."""
        if not self.results:
            print("No results to summarize.")
            return

        print("\n" + "="*60)
        print("SCENARIO COMPARISON SUMMARY")
        print("="*60)

        # Create summary table
        summary_data = []

        for scenario_name, results in self.results.items():
            final_prevalence = results['prevalence'][-1]
            final_incidence = results['incidence'][-1]
            peak_prevalence = max(results['prevalence'])

            # Calculate total cases over simulation period
            total_infectious = sum(results['total_infectious'])

            summary_data.append({
                'Scenario': scenario_name,
                'Final Prevalence': f"{final_prevalence:.1f}",
                'Final Incidence': f"{final_incidence:.1f}",
                'Peak Prevalence': f"{peak_prevalence:.1f}",
                'Avg Infectious': f"{total_infectious/len(results['time']):.1f}"
            })

        # Print table
        df_summary = pd.DataFrame(summary_data)
        print(df_summary.to_string(index=False))
        print("="*60)


def create_baseline_scenarios():
    """Create a set of baseline scenarios for the workshop."""
    runner = ScenarioRunner()

    # Scenario 1: Baseline (no interventions)
    runner.add_scenario(
        name="Baseline",
        parameters={}
    )

    # Scenario 2: Higher transmission
    runner.add_scenario(
        name="High Transmission",
        parameters={
            'beta': 0.08  # 60% higher transmission rate
        }
    )

    # Scenario 3: Lower transmission
    runner.add_scenario(
        name="Low Transmission",
        parameters={
            'beta': 0.03  # 40% lower transmission rate
        }
    )

    # Scenario 4: Faster treatment
    runner.add_scenario(
        name="Enhanced Treatment",
        parameters={
            'treatment_rate': 0.02  # Double the treatment rate
        }
    )

    # Scenario 5: Better recovery
    runner.add_scenario(
        name="Improved Recovery",
        parameters={
            'gamma': 0.01  # Double the natural recovery rate
        }
    )

    return runner


def create_intervention_scenarios():
    """Create scenarios with different intervention combinations."""
    runner = ScenarioRunner()

    # Base scenario
    runner.add_scenario("No Interventions")

    # Vaccination only
    runner.add_scenario(
        name="Vaccination Program",
        parameters={
            'vaccine_efficacy': 0.8  # Assume vaccination is implemented
        }
    )

    # Treatment enhancement only
    runner.add_scenario(
        name="Treatment Enhancement",
        parameters={
            'treatment_rate': 0.025  # 2.5x increase
        }
    )

    # Combined interventions
    runner.add_scenario(
        name="Combined Interventions",
        parameters={
            'vaccine_efficacy': 0.8,
            'treatment_rate': 0.025,
            'beta': 0.035  # Also reduce transmission through other measures
        }
    )

    return runner


def workshop_exercise_1():
    """
    Workshop Exercise 1: Basic parameter exploration

    Participants modify transmission parameters and observe effects.
    """
    print("WORKSHOP EXERCISE 1: Parameter Exploration")
    print("=" * 50)

    runner = create_baseline_scenarios()
    runner.run_all_scenarios(years=5)
    runner.compare_scenarios(metrics=['prevalence', 'incidence'])
    runner.print_scenario_summary()

    return runner


def workshop_exercise_2():
    """
    Workshop Exercise 2: Intervention comparison

    Participants explore different intervention strategies.
    """
    print("\nWORKSHOP EXERCISE 2: Intervention Comparison")
    print("=" * 50)

    runner = create_intervention_scenarios()
    runner.run_all_scenarios(years=10)
    runner.compare_scenarios(metrics=['prevalence', 'incidence'])
    runner.print_scenario_summary()

    return runner


def custom_scenario_template():
    """
    Template for participants to create their own scenarios.

    This function serves as a starting point for custom analysis.
    """
    print("\nCUSTOM SCENARIO TEMPLATE")
    print("=" * 30)
    print("Modify this function to create your own scenarios!")

    runner = ScenarioRunner()

    # TODO: Add your custom scenarios here
    # Example:
    # runner.add_scenario(
    #     name="My Custom Scenario",
    #     parameters={
    #         'beta': 0.06,
    #         'treatment_rate': 0.015,
    #         # Add more parameters as needed
    #     }
    # )

    # runner.run_all_scenarios(years=8)
    # runner.compare_scenarios()
    # runner.print_scenario_summary()

    print("Template ready for customization!")
    return runner


def main():
    """Main function to run workshop exercises."""
    print("TB Model Scenario Analysis")
    print("=" * 30)
    print("This script runs different TB modeling scenarios.")
    print("Modify the scenarios to explore different interventions!\n")

    # Run workshop exercises
    exercise1_runner = workshop_exercise_1()
    exercise2_runner = workshop_exercise_2()

    # Save results
    exercise1_runner.save_comparison_data('../results/exercise1_results.csv')
    exercise2_runner.save_comparison_data('../results/exercise2_results.csv')

    # Template for custom work
    custom_scenario_template()

    print("\nAll exercises completed!")
    print("Check the 'results' folder for output files.")


if __name__ == "__main__":
    main()