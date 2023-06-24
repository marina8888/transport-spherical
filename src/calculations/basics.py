import pandas as pd
import numpy as np

def find_y(experimental_data):
    """
    Return list of y values from a dataset based on the fact that each and all y values should have a 'y Er' column
    @param experimental_data:
    @return:
    """
    exp_df = pd.DataFrame
    if isinstance(experimental_data, pd.DataFrame):
        exp_df = experimental_data
    elif isinstance(experimental_data, str) and experimental_data.endswith('csv'):
        exp_df = pd.read_csv(experimental_data)
    else:
        raise Exception('cannot recognise the experimental data input type')

    y_vals = [x for x in exp_df.columns if 'Er' in x]
    y_vals.remove("phi Er")
    y_vals = [x.replace(" Er", "") for x in y_vals]

    return y_vals

def x_err_to_y_err(experimental_data, y_vals:list):
    """
    @param experimental_data:
    @param y_vals:
    @return:
    """
    exp_df = pd.DataFrame
    if isinstance(experimental_data, pd.DataFrame):
        exp_df = experimental_data
    elif isinstance(experimental_data, str) and experimental_data.endswith('csv'):
        exp_df = pd.read_csv(experimental_data)
    else:
        raise Exception('cannot recognise the experimental data input type')
    exp_df["min_phi"] = exp_df["phi"] + exp_df["phi Er"]
    exp_df["max_phi"] = exp_df["phi"] - exp_df["phi Er"]

    for y in y_vals:
        # Use polyfit from NumPy to calculate the trend for exp_df[y] and exp_df["phi"]
        coefficients_y = np.polyfit(exp_df["phi"], exp_df[y], 6)
        trend_y = np.polyval(coefficients_y, exp_df["phi"])

        # Calculate the new Er vals:
        print(exp_df[f"{y} Er"])
        exp_df[f"{y}_x_er_Er"] = abs(np.polyval(coefficients_y, exp_df["min_phi"]) - np.polyval(coefficients_y, exp_df["max_phi"]))/2
        exp_df[f"{y} Er"] = np.sqrt((exp_df[f"{y} Er"] ** 2) + (exp_df[f"{y}_x_er_Er"] ** 2))
        print(exp_df[f"{y} Er"])
    return exp_df

