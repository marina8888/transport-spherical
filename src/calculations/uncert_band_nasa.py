import src.settings.config_loader as config
import os
import cantera as ct

# using a set of input reaction mechanisms, calculate the uncertainty bands of NASA-7 polynomials.

def calculate(mech_list):
    mech_list = {f"{config.INPUT_DIR_MECH}/{m}" for m in mech_list}

    # use the first mechanism as the comparison point:
    print(mech_list)
    gas = ct.Solution(str(mech_list[0]))
    print(gas)
    for g in gas.species_names():
        # List of polynomial coefficients (example)
        coefficients_list = [[2, -3, 1], [1, 0, -1], [-1, 2, 0]]

        # Evaluate polynomials for a range of x-values
        x_values = np.linspace(-10, 10, 100)
        y_values_list = [np.polyval(coefficients, x_values) for coefficients in coefficients_list]

        # Flatten the list of y-values
        y_values = np.concatenate(y_values_list)

        # Calculate variance of y-values
        variance = np.var(y_values)

        # Compute uncertainty range
        maximum_value = np.max(y_values)
        uncertainty_percentage = np.sqrt(variance) / maximum_value * 100

        print("Uncertainty Range on Y-axis:", uncertainty_percentage)
