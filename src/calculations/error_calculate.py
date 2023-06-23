import pandas as pd
from src.settings.filepaths import output_dir_numerical, input_dir
import os
from src.settings.logger import LogConfig
import pandas as pd

# this file calculates the error between the numerical and experimental files

logger = LogConfig.configure_logger(__name__)



class ErrorCalculator:
    def __init__(self, name_of_numerical_folder: str, name_of_exp_file: str, flame_type: str):

        logger.info(f"Calculating error on experiment results file: {name_of_exp_file}")
        logger.info(f"Using flame type: {flame_type}")
        self.exp_results_file = f"{input_dir}/{name_of_exp_file}"
        self.flame_type = flame_type
        self.numerical_folder_path = f"{output_dir_numerical}/{name_of_numerical_folder}"
        self.exp_df = pd.read_csv(self.exp_results_file)
        self.error = {}
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

        if self.flame_type == "stagnation":
            # define the species that are compared by checking which ones are in the exp file:
            species = [x for x in self.exp_df.columns if 'Er' in x]
            species.remove("phi Er")
            species = [x.replace(" Er", "") for x in species]
            self.exp_df.columns = ['exp_' + col if col in species else col for col in df.columns]
            merged_df = pd.merge(self.exp_df, numerical_df, on=['T_in', 'P', 'U', 'T', 'phi', 'blend'], how = 'inner')

            # Perform the calculations and store in 'error_val' columns
            for i, sp in enumerate(species):
                exp_col = sp + '_exp'
                er_col = sp + ' Er'
                error_val_col = 'error_val' + str(i + 1)
                merged_df[error_val_col] = ((merged_df[sp] - merged_df[exp_col]) / merged_df[er_col]) ** 2

            # Sum the 'error_val' columns to get 'error_val' column
            merged_df['error_val'] = merged_df[[col for col in merged_df.columns if col.startswith('error_val')]].sum(
                axis=1)
            N = len(merged_df)
            N_fsd = len(species)
            return (1 / N) * (1 / N_fsd) * merged_df['error_val'].sum()


        elif self.flame_type == "freely_prop":
            self.exp_df.rename(columns={'flame_speed': 'flame_speed_exp'}, inplace=True)
            merged_df = pd.merge(self.exp_df, numerical_df, on=['T_in', 'P', 'T_in', 'phi', 'blend'], how = 'inner')
            merged_df['error_val'] = ((merged_df['flame_speed'] - merged_df['flame_speed_exp'])/merged_df['flame_speed Er'])**2
            N = len(merged_df)
            return (1 / N) * merged_df['error_val'].sum()

        else:
            raise Exception('cannot recognise flame type')
