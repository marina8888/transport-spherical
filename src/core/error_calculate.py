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
        self.exp_results_file = f"{input_dir}/{name_of_exp_file}"
        self.flame_type = flame_type
        self.numerical_folder_path = f"{output_dir_numerical}/{name_of_numerical_folder}"
        self.exp_df = pd.read_csv(self.exp_results_file)

    def calculate_error_main(self):
        for root, directories, files in os.walk(self.numerical_folder_path):
            for file_name in files:
                mech_name = os.path.splitext(file_name)[0].rsplit('_', 1)[-1]
                file_path = os.path.join(root, file_name)
                numerical_df = pd.read_csv(file_path)
                self.calculate_error(mech_name, numerical_df)

    def calculate_error(self, mech_name: str, numerical_df: pd.DataFrame):
        print(mech_name)
        print(self.exp_df)
        print(numerical_df)

        # get equation constants based on numerical file:
        N = len(numerical_df)

        if self.flame_type == 'stagnation':
            # lookup to the same condition (T_in, P, U, T_in, phi, blend):
            pass


        elif self.flame_type == 'freely_prop_0.5H2_0.5NH3':
            # lookup to the same condition (T_in, P, phi, blend):
            pass
