import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

from src.calculations.basics import make_linestyle, split_df
import src.settings.config_loader as config

# COL_LIST = ["NO", "NH3", "H2", "NH2", "NH", "H"]
# MULT_LIST =  [50, 1, 1.5, 150, 100, 150, 150]
# COLOUR_LIST = ["b", "green", "goldenrod", "darkorange", "red", "mediumpurple"]
# TEXT_SIZE = 16
COL_LIST = ["NO2", "N2O"]
MULT_LIST =  [1500, 100]
COLOUR_LIST = ["red", "darkorange", "gold", "limegreen", "darkgreen", "b"]
# COLOUR_LIST = ["black", "black", "black", "black"]
TEXT_SIZE = 16
LINESTYLE = ['-.', ':', '-']

def plotter_domain(numerical_folder: str):
    """
    Domain emissions for a specific condition
    @param numerical_folder: csv folder for a file with entire numerical domain profiles printed
    @return:
    """
    fig, ax1 = plt.subplots()
    df = pd.read_csv(f"{config.OUTPUT_DIR_NUMERICAL}/{numerical_folder}")
    linestyle = make_linestyle(col_list=COL_LIST)

    ax2 = ax1.twinx()
    colour_list = ["b", "green", "goldenrod", "darkorange", "red", "mediumpurple"]
    for c, col, mult, l in zip(colour_list, COL_LIST, MULT_LIST, linestyle):
        label = col
        if mult != 1:
            label = rf"{col} $ \times $ {mult}"
        ax1.plot(df["grid"], df[col] * mult, color=c, linestyle=l, linewidth=1.5, label=label if l == "-" else None)
    ax1.set_ylim(0, 0.3)
    ax2.set_ylim(290, 3000)
    ax1.set_xlim(0.002, 0.02)
    for col, l in zip(COL_LIST, linestyle):
        # ax2.plot(df["grid"], df["HRR"] * 0.000001, linestyle=l, color="black", label="HRR x-axis location" if l == "-" else None,)
        ax2.plot(df["grid"],df["T"], color = "magenta", linestyle=l, label="Temperature" if l == "-" else None)
        ax1.plot(df["grid"],(df["O"] * 100) + (df["OH"] * 100), "maroon",linestyle=l,label=r"O+OH $ \times $ 100" if l == "-" else None)
    ax1.set_xlabel("flame domain, (m)")
    ax1.set_ylabel(r"Mole fraction, X")
    ax2.set_ylabel(r"Temperature, T (K)")
    # HRR$ \times $$10^{-5}$ (W/$m^{3}$

    # Setting the number of ticks
    ax1.locator_params(axis="both", nbins=4)
    ax2.locator_params(axis="both", nbins=4)
    plt.xlabel(r"grid, m")

    ax1.legend(loc=2)
    ax2.legend(loc=1)
    plt.tight_layout()
    plt.show()

def plotter_domain_sheet(numerical_sheet: str, LABELS_LIST:list):
    """
    Domain emissions for a set of conditions when results are all on one sheet
    @param numerical_sheet:
    @return:
    """
    fig, ax1 = plt.subplots(figsize=(6, 5))
    df = pd.read_csv(numerical_sheet)

    df_split_list = split_df(df = df, labels = LABELS_LIST)
    # ax2 = ax1.twinx()
    i = 0
    for df_split, c, lab in zip(df_split_list, COLOUR_LIST, LABELS_LIST):
        for l, col, mult in zip(LINESTYLE, COL_LIST, MULT_LIST):
            if i == 0:
                label = f"E_NH3 = {lab}"
                # if mult != 1:
                #     label = rf"{col} $ \times $ {mult}"
                i = 1
            else:
                label = None
            ax1.plot(df_split["grid"], df_split[col] * mult, color=c, linestyle=l, linewidth=1.5, label=label)
        i = 0

    ax1.set_ylim(0, 0.6)
    # ax2.set_ylim(290, 3000)
    # ax1.set_xlim(0.002, 0.02)
    ax1.set_xlim(0.0, 0.02)
    ax1.set_xlabel("flame domain, (m)", size = TEXT_SIZE)
    ax1.set_ylabel(r"Mole fraction, X", size = TEXT_SIZE)
    # ax2.set_ylabel(r"Temperature, T (K)")
    # HRR$ \times $$10^{-5}$ (W/$m^{3}$

    # Setting the number of ticks
    ax1.locator_params(axis="both", nbins=4)
    # ax2.locator_params(axis="both", nbins=4)
    plt.xlabel(r"grid, m")
    ax1.legend(fontsize = 8)
    # ax2.legend(loc=1)
    plt.tight_layout()
    plt.figure(figsize=(7, 6), dpi=80)
    plt.show()

def plotter_single(numerical_folder: str, x_col:str, y_col:str,  y_label:str, x_label:str, legend:list = None, num_mulitplier = 1.0 ):
    """
    Plot a single numerical file.
    @param numerical_folder:
    @param exp_results:
    @param col:
    @param exp_multiplier: multiplier for the experimental file
    @param y_label:
    @param num_mulitplier: multiplier for the numerical file
    @return:
    """
    numerical_folder = f"{config.OUTPUT_DIR_NUMERICAL}/{numerical_folder}"
    files = [f for f in os.listdir(numerical_folder) if not f.startswith('.')]
    if legend == None:
        legend = [f.strip('.csv') for f in files]

    # create a linestyle list to loop so the linestyle is always different:
    linestyle = ["--", ":", "-."]
    repetitions = len(files) // len(linestyle)
    remainder =  len(files) % len(linestyle)
    new_list = linestyle * repetitions + linestyle[:remainder]
    linestyle = [new_list[i % len(linestyle)] for i in range(len(files))]

    fig, ax1 = plt.subplots(figsize=(6, 5))
    for file, l, leg in zip(files, linestyle, legend):
        df = pd.read_csv(f"{numerical_folder}/{file}")
        coefficients_y = np.polyfit(df[x_col], df[y_col], 6)
        trend_y = np.polyval(coefficients_y, np.linspace(df[x_col].min(), df[x_col].max(), 20))
        plt.plot(np.linspace(df[x_col].min(), df[x_col].max(),20), trend_y* num_mulitplier,linestyle=l, linewidth=3, label=leg)
        plt.scatter(df[x_col], df[y_col]*num_mulitplier)
        plt.xlabel(x_label, fontsize=TEXT_SIZE)
        plt.ylabel(f"{y_label}", fontsize=TEXT_SIZE)
    plt.tick_params(axis="both", which="major", labelsize=TEXT_SIZE)
    plt.scatter(df[x_col], df[y_col] * num_mulitplier)
    plt.legend(fontsize=TEXT_SIZE)
    plt.ylim(0)
    # plt.xlim(0.88, 1.15)
    plt.tight_layout()
    plt.figure(figsize=(7, 6), dpi=80)
    plt.savefig(f"{config.GRAPH_DIR}/CH4_H2/30%_{y_col}.jpg")
    plt.show()

def plotter_single_input(input_folder: str, x_col: str, y_col: str, x_label: str, y_label: str, legend: list = None,
                   num_mulitplier=1.0):
    """
    Plot a single numerical file.
    @param numerical_folder:
    @param exp_results:
    @param col:
    @param exp_multiplier: multiplier for the experimental file
    @param y_label:
    @param num_mulitplier: multiplier for the numerical file
    @return:
    """
    file = f"{config.INPUT_DIR_NUMERICAL}/{input_folder}"
    df = pd.read_csv(file)
    # df = temp_df.iloc[0:10]
    print(df[x_col])
    y_col_list = ['X_NO2', 'X_N2O', 'X_CO', 'X_NH3', 'X_HCN', 'X_NO']
    # df += temp_df.iloc[-1]
    colours = ['red', 'yellow', 'goldenrod','pink', 'green','blue']
    for y, color in zip(y_col_list, colours):
        print(df[y])
        plt.errorbar(df[x_col], abs(df[y] - df.loc[0, y] * num_mulitplier), yerr=df[f'delta_{y}'], color=color,
                     ls='none')
        plt.scatter(df[x_col], abs(df[y] - df.loc[0, y] * num_mulitplier), label = y, c = color)
    plt.xlabel(x_label, fontsize=TEXT_SIZE)
    plt.ylabel(y_label, fontsize=TEXT_SIZE)
    plt.tick_params(axis="both", which="major", labelsize=TEXT_SIZE)
    plt.legend(fontsize=TEXT_SIZE)
    plt.ylim(0)
    plt.tight_layout()
    plt.figure(figsize=(7, 6), dpi=80)
    plt.show()

