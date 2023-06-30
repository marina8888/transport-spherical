import pandas as pd
import matplotlib.pyplot as plt
from src.settings.filepaths import output_dir

def plot_rop(df: pd.DataFrame, phi:float, blend:str, mech_name:str):

    plt.tight_layout()
    plt.savefig()
    plt.show()
    df.to_csv(f"{output_dir}/rops/{phi}.csv")