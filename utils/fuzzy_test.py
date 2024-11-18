import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyControlSystem:
    def __init__(self, X: pd.DataFrame, y: pd.Series, num_input_mfs: int):
        """
        Initialize the Fuzzy Control System.

        Parameters:
        - X: Pandas DataFrame containing input features.
        - y: Pandas Series containing output values.
        - num_input_mfs: Number of membership functions for each input.
        """
        self.X = X
        self.y = y
        self.num_input_mfs = num_input_mfs

        # Define universes for inputs and outputs
        self.input_universe = np.linspace(X.min().min(), X.max().max(), 100)
        self.output_universe = np.linspace(y.min(), y.max(), 100)

        # Create Antecedents (input variables)
        self.antecedents = [self.define_input_mfs(i) for i in range(X.shape[1])]

        # Create Consequent (output variable)
        self.output = self.define_output_mfs()

        # Initialize rules and weights
        self.rules = []
        self.weights = np.zeros(
            (len(self.antecedents) * self.num_input_mfs, len(self.output))
        )

    def define_input_mfs(self, index: int):
        """
        Define input membership functions for a given input index.

        Parameters:
        - index: Index of the input feature.

        Returns:
        - Antecedent variable with defined MFs.
        """
        antecedent = ctrl.Antecedent(self.input_universe, f"input_{index}")
        antecedent.automf(
            self.num_input_mfs, variable_type="quant"
        )  # Automatically generate membership functions
        return antecedent

    def define_output_mfs(self):
        """
        Define output membership functions.

        Returns:
        - Consequent variable with defined MFs.
        """
        output = ctrl.Consequent(self.output_universe, "output")

        # Correctly access min and max from the output universe
        output_min = output.universe.min()
        output_max = output.universe.max()

        # Define output membership functions
        output["low"] = fuzz.trimf(
            output.universe,
            [
                output_min,
                output_min + (output_max - output_min) / 3,
                output_min + (output_max - output_min) / 3,
            ],
        )

        output["medium"] = fuzz.trimf(
            output.universe,
            [
                output_min + (output_max - output_min) / 3,
                np.mean(output.universe),
                output_max - (output_max - output_min) / 3,
            ],
        )

        output["high"] = fuzz.trimf(
            output.universe,
            [output_max - (output_max - output_min) / 3, output_max, output_max],
        )

        return output

    def define_rules(self):
        """
        Define rules based on input membership functions.

        Returns:
        - List of rules defined in the fuzzy system.
        """
        rules = []

        for i in range(len(self.antecedents)):
            for j in self.antecedents[i].terms:
                for k in self.output.terms:
                    rule = ctrl.Rule(
                        self.antecedents[i][j], self.output[k]
                    )  # Example rule
                    rules.append(rule)

        return rules

    def learn_rules(self, X, y):
        """
        Learn weights for each rule based on the outputs activated by the inputs.

        This is a placeholder learning method that assigns pseudo weights to each rule.

        Returns:
        - Weights matrix indicating activation strength of each rule.
        """
        for idx, features in enumerate(X):

            # Placeholder learning mechanism
            for i in range(len(self.rules)):
                # Here we would normally compute weights based on some learning algorithm
                # For demonstration purposes, we assign random weights
                self.weights[i] = np.random.rand(len(self.output))
            print(self.weights)

    # def set_inputs(self, inputs: dict):
    #     """
    #     Set inputs for the fuzzy control system.

    #     Parameters:
    #     - inputs: Dictionary where keys are input variable names and values are their corresponding values.
    #     """
    #     for key, value in inputs.items():
    #         if key in self.simulation.input:
    #             self.simulation.input[key] = value
    #         else:
    #             raise ValueError(f"Input '{key}' is not recognized.")

    # def compute_output(self):
    #     """Compute the output based on the current inputs."""
    #     self.simulation.compute()

    # def run_simulation(self, inputs: dict):
    #     """Run the simulation with given inputs and return computed output."""

    #     # Set inputs
    #     self.set_inputs(inputs)

    #     # Compute the result
    #     self.compute_output()

    #     return self.simulation.output["output"]


# Example usage:
if __name__ == "__main__":
    # Sample data
    X = pd.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "feature3": [4, 5, 6],
            "feature4": [4, 5, 6],
            "feature5": [4, 5, 6],
        }
    )
    y = pd.Series([0.5, 0.7, 0.9])

    # Create the fuzzy control system with 3 input membership functions
    fuzzy_system = FuzzyControlSystem(X, y, num_input_mfs=3)

    # Define rules after initializing the system
    fuzzy_system.rules = fuzzy_system.define_rules()

    # Learn weights for rules (placeholder implementation)
    fuzzy_system.learn_rules()

    # Set inputs for simulation (using first row of X as an example)
    inputs_to_set = {f"input_{i}": X.iloc[0][i] for i in range(X.shape[1])}

    # Run the simulation and get the computed output
    computed_output = fuzzy_system.run_simulation(inputs_to_set)

    # Output result
    print("Computed Output:", computed_output)
