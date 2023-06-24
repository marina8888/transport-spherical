import pandas as pd
import matplotlib.pyplot as plt
import os
from src.settings.filepaths import output_dir_numerical, input_dir

TEXT_SIZE = 14
def plotter(numerical_folder: str, exp_results: str, col:str, multiplier:float, y_label:str, global_multiplier = 1.0):
    """
    Plot a comparison from a df_numerical_list to experimental comparison along phi on the x axis.
    @param numerical_list: list of csv files to numerical results
    @param exp_results: csv file
    @param col: column to look up on both sheets
    @param multiplier: multiply only the numerical results
    @param y_label: y axis label including units
    @return:
    """
    exp_df = pd.read_csv(f"{input_dir}/{exp_results}")
    numerical_folder = f"{output_dir_numerical}/{numerical_folder}"
    files = os.listdir(numerical_folder)

    #
    linestyle = ["--", ":", "-."]
    repetitions = len(files) // len(linestyle)
    remainder =  len(files) % len(linestyle)
    new_list = linestyle * repetitions + linestyle[:remainder]
    linestyle = [new_list[i % len(linestyle)] for i in range(len(files))]

    for file, l in zip(files, linestyle):
        df = pd.read_csv(f"{numerical_folder}/{file}")
        exp_df = exp_df[exp_df["blend"] == df.loc[0, "blend"]]
        legend = file.split('_')[-1].split('.')[0]
        plt.plot(df["phi"], df[col] * multiplier * global_multiplier, linestyle=l, linewidth=3, label=legend)
        plt.xlabel(r"equivalence ratio, $\phi$", fontsize=TEXT_SIZE)
        plt.ylabel(f"{y_label}", fontsize=TEXT_SIZE)
    plt.scatter(
        exp_df["phi"],
        exp_df[col] * global_multiplier,
        marker="o",
        color="black",
        s=80,
        facecolors="none",
        label="Experiment",
    )
    plt.errorbar(exp_df["phi"], exp_df[col]*global_multiplier, yerr=exp_df[f"{col} Er"]*global_multiplier, color="black", fmt="")
    plt.tick_params(axis="both", which="major", labelsize=TEXT_SIZE)
    plt.legend(fontsize=TEXT_SIZE)
    plt.tight_layout()
    plt.show()

