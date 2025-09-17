"""
TB Transmission Model - SEIR Framework
=====================================

A simplified tuberculosis epidemiological model for teaching Git/GitHub workflows.
This model implements a basic SEIR (Susceptible-Exposed-Infectious-Recovered) framework
with age stratification and intervention capabilities.

Author: Workshop Team
License: Educational Use
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
import os


class TBModel:
    """
    TB SEIR transmission model with age stratification.

    Compartments:
    - S: Susceptible
    - E: Exposed (latent TB)
    - I: Infectious (active TB)
    - R: Recovered/Treated
    """

    def __init__(self, config_file='../data/parameters.yaml'):
        """Initialize model with parameters from config file."""
        self.load_parameters(config_file)
        self.setup_population()
        self.results = None

    def load_parameters(self, config_file):
        """Load model parameters from YAML configuration file."""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            self.params = config['model_parameters']
            self.age_groups = config['age_groups']
            self.time_params = config['time_parameters']

            print(f"Parameters loaded from {config_file}")

        except FileNotFoundError:
            print(f"Config file {config_file} not found. Using default parameters.")
            self.set_default_parameters()

    def set_default_parameters(self):
        """Set default model parameters if config file is not available."""
        self.params = {
            'beta': 0.05,           # Transmission rate per contact per day
            'sigma': 0.0002,        # Progression rate (latent to active)
            'gamma': 0.005,         # Recovery rate
            'mu': 0.00004,          # Natural death rate
            'treatment_rate': 0.01,  # Treatment initiation rate
            'vaccine_efficacy': 0.7  # Vaccine effectiveness
        }

        self.age_groups = ['0-14', '15-44', '45-64', '65+']

        self.time_params = {
            'start_year': 2020,
            'end_year': 2030,
            'time_step': 1  # days
        }

    def setup_population(self):
        """Initialize population compartments by age group."""
        n_age_groups = len(self.age_groups)

        # Initial population sizes (in thousands)
        total_pop = [500, 800, 400, 200]  # Different sizes for age groups

        # Initial conditions
        initial_infectious = [0.1, 0.2, 0.15, 0.05]  # Initial active TB cases
        initial_latent = [2.0, 5.0, 3.0, 1.0]       # Initial latent TB

        self.population = {}
        for i, age_group in enumerate(self.age_groups):
            self.population[age_group] = {
                'S': total_pop[i] - initial_latent[i] - initial_infectious[i],
                'E': initial_latent[i],
                'I': initial_infectious[i],
                'R': 0.0
            }

    def calculate_force_of_infection(self, infectious_by_age):
        """Calculate age-specific force of infection."""
        # Simplified mixing matrix (homogeneous mixing)
        total_infectious = sum(infectious_by_age.values())
        total_population = sum(
            sum(self.population[age].values())
            for age in self.age_groups
        )

        force_of_infection = {}
        for age_group in self.age_groups:
            # Age-specific susceptibility factors
            susceptibility = {
                '0-14': 1.2,    # Children more susceptible
                '15-44': 1.0,   # Reference group
                '45-64': 0.8,   # Adults less susceptible
                '65+': 1.1      # Elderly more susceptible
            }

            force_of_infection[age_group] = (
                self.params['beta'] *
                susceptibility[age_group] *
                total_infectious / total_population
            )

        return force_of_infection

    def apply_interventions(self, t, compartments, age_group):
        """Apply interventions like vaccination and treatment."""
        interventions = {}

        # Vaccination (reduces susceptibility)
        if hasattr(self, 'vaccination_coverage'):
            vaccination_effect = (
                self.vaccination_coverage.get(age_group, 0) *
                self.params['vaccine_efficacy']
            )
            interventions['vaccination'] = vaccination_effect

        # Treatment (increases recovery)
        treatment_factor = 1.0
        if age_group in ['15-44', '45-64']:  # Focus treatment on working age
            treatment_factor = 1.5

        interventions['treatment_factor'] = treatment_factor

        return interventions

    def derivatives(self, t, y, age_group):
        """Calculate derivatives for ODE system."""
        S, E, I, R = y

        # Get current infectious population for force of infection
        current_infectious = {age: self.population[age]['I'] for age in self.age_groups}
        foi = self.calculate_force_of_infection(current_infectious)

        # Apply interventions
        interventions = self.apply_interventions(t, y, age_group)

        # Natural death rate (same for all compartments)
        mu = self.params['mu']

        # Equations
        dS_dt = -foi[age_group] * S - mu * S
        dE_dt = foi[age_group] * S - self.params['sigma'] * E - mu * E
        dI_dt = (self.params['sigma'] * E -
                self.params['gamma'] * I -
                self.params['treatment_rate'] * interventions.get('treatment_factor', 1.0) * I -
                mu * I)
        dR_dt = (self.params['gamma'] * I +
                self.params['treatment_rate'] * interventions.get('treatment_factor', 1.0) * I -
                mu * R)

        return np.array([dS_dt, dE_dt, dI_dt, dR_dt])

    def run_simulation(self, years=10):
        """Run the TB transmission model simulation."""
        print("Starting TB model simulation...")

        # Time setup
        t_start = 0
        t_end = years * 365  # Convert years to days
        dt = self.time_params['time_step']
        time_points = np.arange(t_start, t_end, dt)

        # Initialize results storage
        results = {
            'time': time_points / 365,  # Convert back to years for output
            'total_population': [],
            'total_susceptible': [],
            'total_latent': [],
            'total_infectious': [],
            'total_recovered': [],
            'incidence': [],
            'prevalence': []
        }

        # Add age-specific results
        for age_group in self.age_groups:
            for compartment in ['S', 'E', 'I', 'R']:
                results[f'{age_group}_{compartment}'] = []

        # Simple Euler integration
        for t in time_points:
            # Calculate derivatives for each age group
            for age_group in self.age_groups:
                current_state = [
                    self.population[age_group]['S'],
                    self.population[age_group]['E'],
                    self.population[age_group]['I'],
                    self.population[age_group]['R']
                ]

                derivatives = self.derivatives(t, current_state, age_group)

                # Update populations
                self.population[age_group]['S'] += derivatives[0] * dt
                self.population[age_group]['E'] += derivatives[1] * dt
                self.population[age_group]['I'] += derivatives[2] * dt
                self.population[age_group]['R'] += derivatives[3] * dt

                # Ensure non-negative populations
                for compartment in ['S', 'E', 'I', 'R']:
                    self.population[age_group][compartment] = max(
                        0, self.population[age_group][compartment]
                    )

            # Store results
            total_S = sum(self.population[age]['S'] for age in self.age_groups)
            total_E = sum(self.population[age]['E'] for age in self.age_groups)
            total_I = sum(self.population[age]['I'] for age in self.age_groups)
            total_R = sum(self.population[age]['R'] for age in self.age_groups)
            total_pop = total_S + total_E + total_I + total_R

            results['total_population'].append(total_pop)
            results['total_susceptible'].append(total_S)
            results['total_latent'].append(total_E)
            results['total_infectious'].append(total_I)
            results['total_recovered'].append(total_R)
            results['prevalence'].append(total_I / total_pop * 100000)  # per 100,000

            # Calculate incidence (new active cases)
            new_cases = sum(
                self.params['sigma'] * self.population[age]['E']
                for age in self.age_groups
            )
            results['incidence'].append(new_cases * 365 / total_pop * 100000)  # Annual per 100,000

            # Store age-specific results
            for age_group in self.age_groups:
                for compartment in ['S', 'E', 'I', 'R']:
                    results[f'{age_group}_{compartment}'].append(
                        self.population[age_group][compartment]
                    )

        self.results = results
        print(f"Simulation completed for {years} years")
        return results

    def plot_results(self, save_path='./results/tb_model_output.png'):
        """Create visualization of model results.
        save.path"""

        if self.results is None:
            print("No results to plot. Run simulation first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('TB Model Results', fontsize=16)

        time = self.results['time']

        # Plot 1: Total population by compartment
        axes[0, 0].plot(time, self.results['total_susceptible'], label='Susceptible', color='blue')
        axes[0, 0].plot(time, self.results['total_latent'], label='Latent TB', color='orange')
        axes[0, 0].plot(time, self.results['total_infectious'], label='Active TB', color='red')
        axes[0, 0].plot(time, self.results['total_recovered'], label='Recovered', color='green')
        axes[0, 0].set_title('Population by Compartment')
        axes[0, 0].set_xlabel('Years')
        axes[0, 0].set_ylabel('Population (thousands)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: TB Prevalence
        axes[0, 1].plot(time, self.results['prevalence'], color='red', linewidth=2)
        axes[0, 1].set_title('TB Prevalence')
        axes[0, 1].set_xlabel('Years')
        axes[0, 1].set_ylabel('Cases per 100,000')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: TB Incidence
        axes[1, 0].plot(time, self.results['incidence'], color='darkred', linewidth=2)
        axes[1, 0].set_title('TB Incidence')
        axes[1, 0].set_xlabel('Years')
        axes[1, 0].set_ylabel('New cases per 100,000 per year')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Active TB by age group
        colors = ['blue', 'green', 'orange', 'red']
        for i, age_group in enumerate(self.age_groups):
            axes[1, 1].plot(
                time,
                self.results[f'{age_group}_I'],
                label=age_group,
                color=colors[i],
                linewidth=2
            )
        axes[1, 1].set_title('Active TB Cases by Age Group')
        axes[1, 1].set_xlabel('Years')
        axes[1, 1].set_ylabel('Cases (thousands)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.show()

    def save_results(self, filename='../results/tb_model_results.csv'):
        """Save simulation results to CSV file."""
        if self.results is None:
            print("No results to save. Run simulation first.")
            return

        df = pd.DataFrame(self.results)

        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

    def print_summary(self):
        """Print summary statistics of the simulation."""
        if self.results is None:
            print("No results to summarize. Run simulation first.")
            return

        print("\n" + "="*50)
        print("TB MODEL SIMULATION SUMMARY")
        print("="*50)

        final_year = self.results['time'][-1]
        final_prevalence = self.results['prevalence'][-1]
        final_incidence = self.results['incidence'][-1]

        print(f"Simulation period: {self.results['time'][0]:.1f} - {final_year:.1f} years")
        print(f"Final TB prevalence: {final_prevalence:.1f} per 100,000")
        print(f"Final TB incidence: {final_incidence:.1f} per 100,000 per year")

        print(f"\nFinal population by age group:")
        for age_group in self.age_groups:
            total_age = (
                self.results[f'{age_group}_S'][-1] +
                self.results[f'{age_group}_E'][-1] +
                self.results[f'{age_group}_I'][-1] +
                self.results[f'{age_group}_R'][-1]
            )
            active_cases = self.results[f'{age_group}_I'][-1]
            print(f"  {age_group}: {total_age:.1f}k total, {active_cases:.2f}k active TB")

        # Calculate some key metrics
        peak_prevalence = max(self.results['prevalence'])
        peak_time = self.results['time'][self.results['prevalence'].index(peak_prevalence)]

        print(f"\nPeak prevalence: {peak_prevalence:.1f} per 100,000 at year {peak_time:.1f}")
        print("="*50)


def main():
    """Main function to run the TB model."""
    print("TB Transmission Model - Workshop Version")
    print("======================================")

    # Initialize model
    model = TBModel()

    # Run simulation
    model.run_simulation(years=10)

    # Display results
    model.print_summary()

    # Save outputs
    model.save_results()
    model.plot_results()

    print("\nModel run completed successfully!")
    print("Check the 'results' folder for output files.")


if __name__ == "__main__":
    main()