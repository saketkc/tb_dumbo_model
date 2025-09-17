"""
TB Intervention Modules
======================

This module contains various intervention strategies for TB control,
including vaccination, treatment, and contact reduction measures.

This is designed for workshop participants to practice Git workflows
by adding, modifying, and collaborating on different interventions.
"""

import numpy as np
import yaml


class VaccinationProgram:
    """
    Manages vaccination interventions in the TB model.

    This class handles different vaccination strategies including
    routine childhood vaccination and targeted adult vaccination.
    """

    def __init__(self, config=None):
        """Initialize vaccination program with configuration."""
        self.config = config or self.get_default_config()
        self.coverage = self.config.get('coverage_by_age', {})
        self.efficacy = self.config.get('efficacy', 0.7)
        self.enabled = self.config.get('enabled', False)

    def get_default_config(self):
        """Return default vaccination configuration."""
        return {
            'enabled': False,
            'efficacy': 0.7,
            'coverage_by_age': {
                '0-14': 0.8,
                '15-44': 0.0,
                '45-64': 0.0,
                '65+': 0.0
            },
            'strategy': 'routine'  # 'routine', 'targeted', or 'mass'
        }

    def calculate_protection_factor(self, age_group):
        """
        Calculate protection factor for a given age group.

        Returns the reduction in susceptibility due to vaccination.
        """
        if not self.enabled:
            return 0.0

        coverage = self.coverage.get(age_group, 0.0)
        protection = coverage * self.efficacy

        return protection

    def apply_vaccination(self, susceptible, age_group, dt=1.0):
        """
        Apply vaccination to susceptible population.

        Args:
            susceptible: Number of susceptible individuals
            age_group: Age group identifier
            dt: Time step

        Returns:
            Reduction in susceptible population due to vaccination
        """
        if not self.enabled:
            return 0.0

        coverage = self.coverage.get(age_group, 0.0)
        vaccination_rate = coverage * 0.001  # Daily vaccination rate

        vaccinated = susceptible * vaccination_rate * dt
        return min(vaccinated, susceptible)


class TreatmentProgram:
    """
    Manages treatment interventions for active TB cases.

    Includes different treatment strategies, drug resistance considerations,
    and treatment completion rates.
    """

    def __init__(self, config=None):
        """Initialize treatment program."""
        self.config = config or self.get_default_config()
        self.base_rate = self.config.get('base_treatment_rate', 0.01)
        self.enabled = self.config.get('enabled', True)
        self.drug_resistance_rate = self.config.get('drug_resistance_rate', 0.05)

    def get_default_config(self):
        """Return default treatment configuration."""
        return {
            'enabled': True,
            'base_treatment_rate': 0.01,  # Per day
            'enhanced_rate_multiplier': 2.0,
            'drug_resistance_rate': 0.05,
            'treatment_success_rate': 0.85,
            'age_specific_factors': {
                '0-14': 1.0,
                '15-44': 1.2,  # Better treatment access for working age
                '45-64': 1.2,
                '65+': 0.8     # Lower treatment success in elderly
            }
        }

    def calculate_treatment_rate(self, age_group, enhanced=False):
        """
        Calculate age-specific treatment rate.

        Args:
            age_group: Age group identifier
            enhanced: Whether enhanced treatment program is active

        Returns:
            Treatment rate per day
        """
        if not self.enabled:
            return 0.0

        base_rate = self.base_rate
        age_factor = self.config.get('age_specific_factors', {}).get(age_group, 1.0)

        rate = base_rate * age_factor

        if enhanced:
            multiplier = self.config.get('enhanced_rate_multiplier', 2.0)
            rate *= multiplier

        return rate

    def calculate_treatment_success(self, age_group):
        """Calculate treatment success rate by age group."""
        base_success = self.config.get('treatment_success_rate', 0.85)
        age_factor = self.config.get('age_specific_factors', {}).get(age_group, 1.0)

        # Adjust success rate based on age factor
        success_rate = base_success * min(age_factor, 1.0)
        return min(success_rate, 1.0)


class ContactReduction:
    """
    Models interventions that reduce transmission through contact reduction.

    Examples include social distancing, isolation, improved ventilation,
    and behavioral change campaigns.
    """

    def __init__(self, config=None):
        """Initialize contact reduction intervention."""
        self.config = config or self.get_default_config()
        self.enabled = self.config.get('enabled', False)
        self.reduction_factor = self.config.get('reduction_factor', 0.7)

    def get_default_config(self):
        """Return default contact reduction configuration."""
        return {
            'enabled': False,
            'reduction_factor': 0.7,  # 30% reduction in effective contacts
            'intervention_type': 'behavioral',  # 'behavioral', 'isolation', 'environmental'
            'duration': 365,  # Duration in days
            'compliance_rate': 0.8  # Population compliance
        }

    def calculate_transmission_reduction(self, time_point=0):
        """
        Calculate reduction in transmission coefficient.

        Args:
            time_point: Current time in simulation

        Returns:
            Factor by which to multiply transmission rate (0-1)
        """
        if not self.enabled:
            return 1.0

        duration = self.config.get('duration', 365)
        compliance = self.config.get('compliance_rate', 0.8)

        # Check if intervention is still active
        if time_point > duration:
            return 1.0

        # Calculate effective reduction
        effective_reduction = self.reduction_factor * compliance
        transmission_factor = 1.0 - effective_reduction

        return max(transmission_factor, 0.1)  # Minimum 10% transmission


class ActiveCaseFinding:
    """
    Models active case finding interventions to detect TB cases earlier.

    This intervention increases the detection rate of infectious individuals,
    moving them into treatment faster.
    """

    def __init__(self, config=None):
        """Initialize active case finding program."""
        self.config = config or self.get_default_config()
        self.enabled = self.config.get('enabled', False)
        self.detection_multiplier = self.config.get('detection_multiplier', 2.0)

    def get_default_config(self):
        """Return default active case finding configuration."""
        return {
            'enabled': False,
            'detection_multiplier': 2.0,  # Increase detection rate by factor of 2
            'target_populations': ['all'],  # Can be ['all'], ['high_risk'], ['contacts']
            'frequency': 'annual',  # 'annual', 'continuous'
            'coverage': 0.7  # Proportion of target population reached
        }

    def calculate_detection_rate(self, base_rate, age_group=None):
        """
        Calculate enhanced detection rate due to active case finding.

        Args:
            base_rate: Baseline detection/treatment initiation rate
            age_group: Age group (for targeted interventions)

        Returns:
            Enhanced detection rate
        """
        if not self.enabled:
            return base_rate

        coverage = self.config.get('coverage', 0.7)
        multiplier = self.detection_multiplier

        # Calculate proportion of population reached
        enhanced_rate = base_rate * (1 + (multiplier - 1) * coverage)

        return enhanced_rate


class InterventionManager:
    """
    Coordinates multiple interventions and manages their interactions.

    This class handles the implementation of multiple simultaneous interventions
    and ensures they work together appropriately.
    """

    def __init__(self, config_file=None):
        """Initialize intervention manager."""
        self.interventions = {}

        if config_file:
            self.load_from_config(config_file)
        else:
            self.setup_default_interventions()

    def load_from_config(self, config_file):
        """Load intervention configurations from YAML file."""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            intervention_configs = config.get('interventions', {})

            # Initialize each intervention type
            if 'vaccination' in intervention_configs:
                self.interventions['vaccination'] = VaccinationProgram(
                    intervention_configs['vaccination']
                )

            if 'enhanced_treatment' in intervention_configs:
                self.interventions['treatment'] = TreatmentProgram(
                    intervention_configs['enhanced_treatment']
                )

            if 'contact_reduction' in intervention_configs:
                self.interventions['contact_reduction'] = ContactReduction(
                    intervention_configs['contact_reduction']
                )

            if 'active_case_finding' in intervention_configs:
                self.interventions['case_finding'] = ActiveCaseFinding(
                    intervention_configs['active_case_finding']
                )

        except FileNotFoundError:
            print(f"Config file {config_file} not found. Using defaults.")
            self.setup_default_interventions()

    def setup_default_interventions(self):
        """Set up default intervention configurations."""
        self.interventions = {
            'vaccination': VaccinationProgram(),
            'treatment': TreatmentProgram(),
            'contact_reduction': ContactReduction(),
            'case_finding': ActiveCaseFinding()
        }

    def get_transmission_factor(self, time_point=0):
        """
        Calculate combined effect of interventions on transmission.

        Returns factor by which to multiply transmission coefficient.
        """
        factor = 1.0

        # Apply contact reduction
        if 'contact_reduction' in self.interventions:
            factor *= self.interventions['contact_reduction'].calculate_transmission_reduction(
                time_point
            )

        return factor

    def get_protection_factor(self, age_group):
        """
        Calculate protection from vaccination for given age group.

        Returns reduction in susceptibility due to vaccination.
        """
        if 'vaccination' in self.interventions:
            return self.interventions['vaccination'].calculate_protection_factor(age_group)
        return 0.0

    def get_treatment_rate(self, age_group, base_rate):
        """
        Calculate enhanced treatment rate including all treatment interventions.

        Args:
            age_group: Age group identifier
            base_rate: Base treatment rate from model

        Returns:
            Enhanced treatment rate
        """
        enhanced_rate = base_rate

        # Apply enhanced treatment program
        if 'treatment' in self.interventions:
            enhanced_rate = self.interventions['treatment'].calculate_treatment_rate(
                age_group, enhanced=True
            )

        # Apply active case finding
        if 'case_finding' in self.interventions:
            enhanced_rate = self.interventions['case_finding'].calculate_detection_rate(
                enhanced_rate, age_group
            )

        return enhanced_rate

    def print_intervention_summary(self):
        """Print summary of active interventions."""
        print("\nActive Interventions:")
        print("-" * 30)

        for name, intervention in self.interventions.items():
            if hasattr(intervention, 'enabled') and intervention.enabled:
                print(f"✓ {name.replace('_', ' ').title()}")
            else:
                print(f"✗ {name.replace('_', ' ').title()} (disabled)")

        print("-" * 30)


# Example usage and testing functions
def test_vaccination_program():
    """Test vaccination program functionality."""
    print("Testing Vaccination Program")
    print("=" * 30)

    # Test default configuration
    vacc = VaccinationProgram()
    print(f"Default enabled: {vacc.enabled}")
    print(f"Default efficacy: {vacc.efficacy}")

    # Test with custom configuration
    config = {
        'enabled': True,
        'efficacy': 0.8,
        'coverage_by_age': {
            '0-14': 0.9,
            '15-44': 0.5,
            '45-64': 0.3,
            '65+': 0.6
        }
    }

    vacc_custom = VaccinationProgram(config)

    for age_group in ['0-14', '15-44', '45-64', '65+']:
        protection = vacc_custom.calculate_protection_factor(age_group)
        print(f"Protection for {age_group}: {protection:.2f}")


def test_intervention_manager():
    """Test the intervention manager."""
    print("\nTesting Intervention Manager")
    print("=" * 35)

    manager = InterventionManager()
    manager.print_intervention_summary()

    # Test combined effects
    transmission_factor = manager.get_transmission_factor(time_point=100)
    print(f"\nTransmission factor: {transmission_factor:.3f}")

    for age_group in ['0-14', '15-44', '45-64', '65+']:
        protection = manager.get_protection_factor(age_group)
        treatment_rate = manager.get_treatment_rate(age_group, base_rate=0.01)
        print(f"{age_group}: Protection={protection:.3f}, Treatment rate={treatment_rate:.4f}")


if __name__ == "__main__":
    # Run tests
    test_vaccination_program()
    test_intervention_manager()