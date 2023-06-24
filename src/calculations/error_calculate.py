from src.calculations.basics import find_y, x_err_to_y_err
from src.settings.filepaths import output_dir_numerical, input_dir
import os
from src.settings.logger import LogConfig
import pandas as pd

# this file calculates the error between the numerical and experimental files

logger = LogConfig.configure_logger(__name__)



class ErrorCalculator:
    def __init__(self, name_of_numerical_folder: str, name_of_exp_file: str, flame_type: str):
        """
        Create an instance for every experimental file used such that an error dictionary is created
        @param name_of_numerical_folder:
        @param name_of_exp_file:
        @param flame_type:
        """
        logger.info(f"Calculating error on experiment results file: {name_of_exp_file}")
        logger.info(f"Using flame type: {flame_type}")

        self.exp_results_file = f"{input_dir}/{name_of_exp_file}"
        self.flame_type = flame_type
        self.numerical_folder_path = f"{output_dir_numerical}/{name_of_numerical_folder}"
        self.exp_df = pd.read_csv(self.exp_results_file)
        self.error = {}

        # find the list of y values and use them for the exp df:
        self.y_vals = find_y(self.exp_df)
        self.exp_df = x_err_to_y_err(self.exp_df, self.y_vals) #convert x error into y
        self.exp_df.columns = ['exp_' + col if col in self.y_vals else col for col in self.exp_df.columns]
        self.calculate_error_main()

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

        # merge based on alignment between key input conditions:
        if self.flame_type == "stagnation":
            merged_df = pd.merge(self.exp_df, numerical_df, on=['T_in', 'P', 'U', 'T', 'phi', 'blend'], how = 'inner')
        elif self.flame_type == "freely_prop":
            merged_df = pd.merge(self.exp_df, numerical_df, on=['T_in', 'P', 'T_in', 'phi', 'blend'], how = 'inner')
        else:
            raise Exception('cannot recognise flame type')

        # Perform the calculations and store in 'error_val' columns
        for y in self.y_vals:
            print(f"y vals are: {self.y_vals}")
            merged_df[f"error_val_{y}"] = ((merged_df[y] - merged_df[f"exp_{y}"]) / merged_df[f"{y} Er"]) ** 2

        # Sum the 'error_val' columns to get 'error_val' column
        merged_df['error_val'] = merged_df[[col for col in merged_df.columns if col.startswith('error_val')]].sum(axis=1)

        N = len(merged_df)
        N_fsd = len(self.y_vals)
        error = (1 / N) * (1 / N_fsd) * merged_df['error_val'].sum()

        return error

    def get_error(self):
        return self.error