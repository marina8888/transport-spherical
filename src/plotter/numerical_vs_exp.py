import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import math
import os
import numpy as np

import src.settings.config_loader as config


TEXT_SIZE = 16
def plotter(numerical_folder: str, exp_results: str, col:str, exp_multiplier:float, y_label:str, num_mulitplier = 1.0, title = None):
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
    plt.figure(figsize=(6,6))
    exp_df = pd.read_csv(f"{config.INPUT_DIR_NUMERICAL}/{exp_results}")
    numerical_folder = f"{config.OUTPUT_DIR_NUMERICAL}/{numerical_folder}"
    files =  [f for f in os.listdir(numerical_folder) if not f.startswith('.')]
    # create a linestyle list to loop so the linestyle is always different:
    linestyle = ["--", ":", "-.", "--", ":", "--", ":", "-.", "--", ":", "-."]
    colour = ['green', 'blue', 'orange','red', 'purple', 'skyblue', 'pink', 'lime', 'silver']
    # colour = ['green', 'blue', 'orange', 'red', 'goldenrod', 'purple', 'skyblue', 'pink', 'lime', 'silver']
    for file, l, c in zip(files, linestyle, colour):
        df = pd.read_csv(f"{numerical_folder}/{file}")
        legend = file.split('_')[-1].split('.')[0]
        print(file, l, legend, c)
        plt.plot(df["phi"], df[col] * num_mulitplier, linestyle=l, color = c, linewidth=3, label=legend)
    plt.title(title)
    plt.xlabel(r"equivalence ratio, $\phi$", fontsize=TEXT_SIZE)
    plt.ylabel(f"{y_label}", fontsize=TEXT_SIZE)
    # coefficients_y = np.polyfit(exp_df["phi"], exp_df[col], 21)
    # trend_y = np.polyval(coefficients_y, (np.linspace(exp_df['phi'].min(), exp_df['phi'].max(), 20, endpoint=True)))
    # plt.plot(np.linspace(exp_df['phi'].min(), exp_df['phi'].max(), 20, endpoint=True),
    #     trend_y * exp_multiplier,color="black", linestyle = '-', marker = None )
    plt.plot(exp_df["phi"], exp_df[col] * exp_multiplier, marker = None, linestyle = '-', color="black")
    plt.scatter(exp_df["phi"], exp_df[col] * exp_multiplier, s = 80, marker="o",facecolor='none', linestyle = '', color="black",
        label="Experiment")
    plt.errorbar(exp_df["phi"], exp_df[col]*exp_multiplier, yerr=exp_df[f"{col} Er"]*exp_multiplier, linestyle = '', color="black")
    plt.tick_params(axis="both", which="major", labelsize=TEXT_SIZE)
    plt.legend(fontsize=12)
    plt.ylim(0, 100000)
    plt.yticks([0, 20000, 40000, 60000, 80000, 100000])
    plt.xlim(exp_df['phi'].min()-0.05, exp_df['phi'].max()+0.05)
    # plt.xticks([0.6, 0.7, 0.8, 0.9, 1.0, 1.10, 1.20, 1.30])
    plt.tight_layout()
    plt.savefig(f"{config.GRAPHS_DIR}/tester_{col}.jpg")
    plt.show()

def plot_all(folder, species, multiplier):
    colors = ['grey', 'r', 'blue', 'green', 'purple']
    labels = ['10%', '20%', '30%', '40%', '60%']
    folders = ['10%_data_reduced.csv', '20%_data_reduced.csv', '30%_data_reduced.csv', '40%_data_reduced.csv', '60%_data_reduced.csv']
    for f,c,l in zip(folders, colors, labels):
        df = pd.read_csv(f"{config.INPUT_DIR_NUMERICAL}/{folder}/{f}")
        coefficients_y = np.polyfit(df["phi"], df[species], 6)
        trend_y = np.polyval(coefficients_y, np.linspace(df["phi"].min(), df["phi"].max(), 20))
        plt.plot(np.linspace(df["phi"].min(), df["phi"].max(),20), trend_y* multiplier,label = l, color = c, linewidth=3)
        plt.scatter(df["phi"], df[species] * multiplier, color = c)
        # plt.plot(df["phi"], df[species] * multiplier, color = c, label = df.loc[0,'blend'])

        # plt.plot(df["phi"], df[col] * num_mulitplier, linestyle=l, linewidth=3, label=legend)
    plt.xlabel(r"equivalence ratio, $\phi$", fontsize=TEXT_SIZE)
    plt.ylabel(f"{species}", fontsize=TEXT_SIZE)
    plt.tight_layout()
    plt.figsize = (10,5)
    plt.xlim(0.4, 1.45)
    plt.savefig(f"{config.GRAPHS_DIR}/tester_all.jpg")
    plt.show()
    plt.legend()
    plt.switch_backend('Agg')