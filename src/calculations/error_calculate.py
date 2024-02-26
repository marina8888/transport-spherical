import os
import pandas as pd

from src.calculations.basics import find_y, x_err_to_y_err

from src.settings.logger import LogConfig
import src.settings.config_loader as config


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

        self.exp_results_file = f"{config.INPUT_DIR_NUMERICAL}/{name_of_exp_file}"
        self.flame_type = flame_type
        self.numerical_folder_path = f"{config.OUTPUT_DIR_NUMERICAL}/{name_of_numerical_folder}"
        self.files = [f for f in os.listdir(self.numerical_folder_path) if not f.startswith('.')]
        self.exp_df = pd.read_csv(self.exp_results_file)
        # In this example, we drop rows with missing values in 'key' column


        # find the list of y values and use them for the exp df:
        self.y_vals = find_y(self.exp_df, exclude_carbon_sp = True, exclude_water = True, exclude_oxygen = False, exclude_minor = True)
        self.error = {}
        self.df_error_sp = pd.DataFrame(columns=self.y_vals)
        print(self.df_error_sp)


        self.exp_df = x_err_to_y_err(self.exp_df, self.y_vals) #convert x error into y
        self.exp_df.columns = ['exp_' + col if col in self.y_vals else col for col in self.exp_df.columns]
        self.calculate_error_main()
        self.df_error_sp.to_csv(f"{config.OUTPUT_DIR_ERROR}/sp_error_{os.path.splitext(self.exp_results_file.split('/')[-1])[0]}.csv")

        print(self.error)

    def calculate_error_main(self):
        for file_name in self.files:
            mech_name = file_name.rsplit(".", 1)[0]
            numerical_df = pd.read_csv(f"{self.numerical_folder_path }/{file_name}")
            self.error[mech_name] = self.calculate_error(numerical_df, mech_name)
            logger.info(f"Calculating error for mechanism file: {mech_name}")
            logger.info(f"Calculating error for numerical file: {self.numerical_folder_path}/{file_name}")


    def calculate_error(self, numerical_df: pd.DataFrame, mech_name):


        # merge based on alignment between key input conditions:
        if self.flame_type == "stagnation":
             merged_df = pd.merge( self.exp_df, numerical_df, on=['P', 'phi'], how = 'inner')
             print(f'length {len(merged_df)}')
        elif self.flame_type == "freely_prop":
            merged_df = pd.merge(self.exp_df, numerical_df, on=['T', 'P','phi'], how = 'inner')
        else:
            raise Exception('cannot recognise flame type')

        if len(merged_df) == 0:
            raise TypeError(f' {mech_name}: Your experimental file does not cover all the numerical file conditions')
        # Perform the calculations and store in 'error_val' columns
        for y in self.y_vals:
            if y == 'O2' or y == 'H2':
                merged_df[f"error_val_{y}"] = ((merged_df[y] - (merged_df[f"exp_{y}"]*0.01)) / (0.01 * merged_df[f"{y} Er"])) ** 2
            else:
                merged_df[f"error_val_{y}"] = ((merged_df[y] - (merged_df[f"exp_{y}"]*0.000001))/ (0.000001 * merged_df[f"{y} Er"])) ** 2
        error_per_sp = merged_df[[col for col in merged_df.columns if col.startswith('error_val')]].sum().tolist()
        error_sp = [value / (len(merged_df)) for value in error_per_sp]
        self.df_error_sp.loc[mech_name] = error_sp
        error = sum(error_sp)/(len(self.y_vals))
        return error

    def get_error(self):
        return self.error