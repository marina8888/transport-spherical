import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from src.settings.filepaths import output_dir_numerical, input_dir, output_dir

TEXT_SIZE = 14
def plotter(numerical_folder: str, exp_results: str, col:str, exp_multiplier:float, y_label:str, num_mulitplier = 1.0):
    """
    Plot a comparison from a df_numerical_list to experimental comparison along phi on the x axis.
    @param numerical_folder:
    @param exp_results:
    @param col:
    @param exp_multiplier: multiplier for the experimental file
    @param y_label:
    @param num_mulitplier: multiplier for the numerical file
    @return:
    """
    exp_df = pd.read_csv(f"{input_dir}/{exp_results}")
    numerical_folder = f"{output_dir_numerical}/{numerical_folder}"
    files = files = [f for f in os.listdir(numerical_folder) if not f.startswith('.')]

    # create a linestyle list to loop so the linestyle is always different:
    linestyle = ["--", ":", "-."]
    repetitions = len(files) // len(linestyle)
    remainder =  len(files) % len(linestyle)
    new_list = linestyle * repetitions + linestyle[:remainder]
    linestyle = [new_list[i % len(linestyle)] for i in range(len(files))]

    for file, l in zip(files, linestyle):
        df = pd.read_csv(f"{numerical_folder}/{file}")
        exp_df = exp_df[exp_df["blend"] == df.loc[0, "blend"]]
        legend = file.split('_')[-1].split('.')[0]
        plt.plot(df["phi"], df[col] * num_mulitplier, linestyle=l, linewidth=3, label=legend)
        plt.xlabel(r"equivalence ratio, $\phi$", fontsize=TEXT_SIZE)
        plt.ylabel(f"{y_label}", fontsize=TEXT_SIZE)
    coefficients_y = np.polyfit(exp_df["phi"], exp_df[col], 6)
    trend_y = np.polyval(coefficients_y, (np.linspace(exp_df['phi'].min(), exp_df['phi'].max(), 30, endpoint=True))
)
    plt.plot(
        np.linspace(exp_df['phi'].min(), exp_df['phi'].max(), 30, endpoint=True),
        trend_y * exp_multiplier,
        color="black", linestyle = '-', marker = None,
    )
    plt.scatter(exp_df["phi"], exp_df[col] * exp_multiplier, s = 80, marker="o",facecolor='none', linestyle = '', color="black",
        label="Experiment")
    plt.errorbar(exp_df["phi"], exp_df[col]*exp_multiplier, yerr=exp_df[f"{col} Er"]*exp_multiplier, linestyle = '', color="black", fmt="")
    plt.tick_params(axis="both", which="major", labelsize=TEXT_SIZE)
    # plt.legend(fontsize=TEXT_SIZE)
    plt.ylim(0)
    plt.xlim(0.58, 1.35)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/graphs/20%/{col}.jpg")
    plt.show()
    plt.switch_backend('Agg')

