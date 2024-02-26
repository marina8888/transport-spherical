import pandas as pd
import numpy as np
import cantera.cti2yaml as cli

def find_y(experimental_data, exclude_carbon_sp = False, exclude_water = False, exclude_oxygen = False, exclude_minor = True):
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
    if exclude_carbon_sp == True:
        carbon_sp = ["CO Er", "C2O Er", "CO2 Er", "CH4 Er"]
        y_vals = [x for x in y_vals if x not in carbon_sp]
    if exclude_water == True:
        water_sp = ["H2O Er"]
        y_vals = [x for x in y_vals if x not in water_sp]
    if exclude_oxygen == True:
        ox_sp = ["O2 Er"]
        y_vals = [x for x in y_vals if x not in ox_sp]
    if exclude_minor == True:
        minor_sp = ["HNCO Er", "HCHO Er", "HCNO Er", "HCN Er"]
        y_vals = [x for x in y_vals if x not in minor_sp]
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

        # Calculate the new Er vals:
        exp_df[f"{y}_x_er_Er"] = abs(np.polyval(coefficients_y, exp_df["min_phi"]) - np.polyval(coefficients_y, exp_df["max_phi"]))/2
        exp_df[f"{y} Er"] = np.sqrt((exp_df[f"{y} Er"] ** 2) + (exp_df[f"{y}_x_er_Er"] ** 2))
        print(exp_df[f"{y} Er"])
    return exp_df

def split_df(df, labels):
    """
    Split up a large dataframe by grid points = 0 so to find individual flames
    @param df:
    @param labels:
    @return:
    """
    split_dfs = []

    # Find the index of the row where "grid" column equals 0
    df =  df.drop("Unnamed: 0", axis=1).reset_index(drop=True)
    split_index = df.index[df["grid"] == 0].to_list()


    while len(split_index) > 1:
        # Split the DataFrame into two based on the split_index
        df_new = df.iloc[split_index[0]:int(split_index[1])-1]

        # remove the index just used and restart
        split_dfs.append(df_new)
        split_index.pop(0)


    # Add the last DataFrame after the last occurrence of "grid" column equals 0
    split_dfs.append(df.iloc[int(split_index[0]):])

    if len(split_dfs) != len(labels):
        raise TypeError("please check the length of your label")
    else:
        return split_dfs

def make_linestyle(col_list:list):
    """
    create a linestyle list to loop so the linestyle is always different:
    @param col_list:
    @return:
    """
    linestyle = ["--", ":", "-."]
    repetitions = len(col_list) // len(linestyle)
    remainder =  len(col_list) % len(linestyle)
    new_list = linestyle * repetitions + linestyle[:remainder]
    linestyle = [new_list[i % len(linestyle)] for i in range(len(col_list))]
    return linestyle
