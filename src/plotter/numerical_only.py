import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os
from src.settings.filepaths import output_dir, output_dir_numerical_output

figure(figsize=(7, 6), dpi=80)
COL_LIST = ["NO", "NH3", "H2", "NH2", "NH", "H"]
MULT_LIST =  [50, 1, 1.5, 150, 100, 150, 150]
TEXT_SIZE = 16

def plotter_domain(numerical_folder: str):
    """
    Domain emissions for a specific condition
    @param numerical_folder: csv folder for a file with entire numerical domain profiles printed
    @return:
    """
    fig, ax1 = plt.subplots()
    df = pd.read_csv(numerical_folder)

    # create a linestyle list to loop so the linestyle is always different:
    linestyle = ["--", ":", "-."]
    repetitions = len(COL_LIST) // len(linestyle)
    remainder =  len(COL_LIST) % len(linestyle)
    new_list = linestyle * repetitions + linestyle[:remainder]
    linestyle = [new_list[i % len(linestyle)] for i in range(len(COL_LIST))]

    ax2 = ax1.twinx()
    colour_list = ["b", "green", "goldenrod", "darkorange", "red", "mediumpurple"]
    for c, col, mult, l in zip(colour_list, COL_LIST, MULT_LIST, linestyle):
        label = col
        if mult != 1:
            label = rf"{col} $ \times $ {mult}"
        ax1.plot(
            df["grid"],
            df[col] * mult,
            color=c,
            linestyle=l,
            linewidth=1.5,
            label=label if l == "-" else None,
            )
    ax1.set_ylim(0, 0.3)
    ax2.set_ylim(290, 3000)
    ax1.set_xlim(0.002, 0.02)
    for df, l in zip(COL_LIST, linestyle):
        ax2.plot(
            df["grid"],
            df["HRR"] * 0.000001,
            linestyle=l,
            color="black",
            label="HRR x-axis location" if l == "-" else None,
        )
        ax2.plot(
            df["grid"],
            df["T"],
            "magenta",
            linestyle=l,
            label="Temperature" if l == "-" else None,
        )
        ax1.plot(
            df["grid"],
            (df["O"] * 100) + (df["OH"] * 100),
            "maroon",
            linestyle=l,
            label=r"O+OH $ \times $ 100" if l == "-" else None,
        )
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


def plotter_single(numerical_folder: str, x_col:str, y_col:str,  y_label:str, x_label:str, legend:list, num_mulitplier = 1.0, ):
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
    numerical_folder = f"{output_dir_numerical_output}/{numerical_folder}"
    files = files = [f for f in os.listdir(numerical_folder) if not f.startswith('.')]

    # create a linestyle list to loop so the linestyle is always different:
    linestyle = ["--", ":", "-."]
    repetitions = len(files) // len(linestyle)
    remainder =  len(files) % len(linestyle)
    new_list = linestyle * repetitions + linestyle[:remainder]
    linestyle = [new_list[i % len(linestyle)] for i in range(len(files))]

    for file, l, leg in zip(files, linestyle, legend):
        df = pd.read_csv(f"{numerical_folder}/{file}")
        df[x_col] = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        coefficients_y = np.polyfit([0, 0.2, 0.4, 0.6, 0.8, 1.0], df[y_col], 3)
        trend_y = np.polyval(coefficients_y, np.linspace(0, 1, 20))
        plt.plot(np.linspace(0, 1, 20), trend_y * num_mulitplier, linestyle=l, linewidth=3, label=leg)
        plt.xlabel(x_label, fontsize=TEXT_SIZE)
        plt.ylabel(f"{y_label}", fontsize=TEXT_SIZE)
    plt.tick_params(axis="both", which="major", labelsize=TEXT_SIZE)
    plt.legend(fontsize=TEXT_SIZE)
    # plt.ylim(0)
    # plt.xlim(0.88, 1.15)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/graphs/CH4_NH3/{y_col}.jpg")
    plt.show()
    plt.switch_backend('Agg')