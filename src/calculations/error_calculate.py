import pandas as pd
from src.settings.filepaths import output_dir_numerical, input_dir
import os
from src.settings.logger import LogConfig
from tqdm import tqdm

# this file calculates the error between the numerical and experimental files

logger = LogConfig.configure_logger(__name__)

import os
import pandas as pd

class ErrorCalculator:
    def __init__(self, name_of_exp_file: str, name_of_numerical_folder: str, flame_type: str):

        logger.info(f"Calculating error on experiment results file: {name_of_exp_file}")
        logger.info(f"Using flame type: {flame_type}")
        self.exp_results_file = f"{input_dir}/{name_of_exp_file}"
        self.flame_type = flame_type
        self.numerical_folder_path = f"{output_dir_numerical}/{name_of_numerical_folder}"
        self.exp_df = pd.read_csv(self.exp_results_file)
        self.error = {}

    def calculate_error_main(self):
        for root, directories, files in os.walk(self.numerical_folder_path):
            for file_name in files:
                mech_name = os.path.splitext(file_name)[0].rsplit('_', 1)[-1]
                file_path = os.path.join(root, file_name)
                numerical_df = pd.read_csv(file_path)

                logger.info(f"Error for mechanism file: {mech_name}")
                logger.info(f"Error for numerical file: {file_path}")

                self.error[mech_name] = self.calculate_error(numerical_df)

    def calculate_error(self, numerical_df: pd.DataFrame):

        # get equation constants based on numerical file:
        N = len(numerical_df)

        if self.flame_type == "stagnation":
            # self.exp_df
            # lookup to the same condition (T_in, P, U, T_in, phi, blend):
            pass


        elif self.flame_type == "freely_prop":
            # lookup to the same condition (T_in, P, phi, blend):
            pass

        return error_value