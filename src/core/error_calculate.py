import pandas as pd
from src.settings.filepaths import output_dir_numerical
import os
from pandas as pd
from src.settings.logger import LogConfig
from tqdm import tqdm

# this file calculates the error between the numerical and experimental files

logger = LogConfig.configure_logger(__name__)

def calculate_error_main(exp_results:str, name_of_numerical_folder):
    numerical_folder_path = f"{output_dir_numerical}/{name_of_numerical_folder}"
    exp_df = pd.read_csv(exp_results)

    for root, directories, files in os.walk(numerical_folder_path):
        for file_name in files:
            mech_name = os.path.splitext(file_name)[0].rsplit('_', 1)[-1]
            file_path = os.path.join(root, file_name)
            numerical_df = pd.read_csv(file_path)
            calculate_error(mech_name, exp_df, numerical_df)


def calculate_error(mech_name:str, exp_df: pd.DataFrame, numerical_df: pd.DataFrame):
    print(mech_name)
    print(exp_df)
    print(numerical_df)