import pandas as pd
from src.settings.filepaths import input_dir, output_dir, mech_dir
from src.flames.stagnation_flame import StagnationFlame
from src.flames.freely_prop_flame import FreelyPropFlame
import os
from src.settings.logger import LogConfig
from tqdm import tqdm

# this file calls various flame types to create numerical csv solution files

logger = LogConfig.configure_logger(__name__)

def run_flame(mech:str, exp_results:str, flame_type:str):
    """

    :param mech:
    :param exp_results:
    :param flame_type: stagnation, freely_prop_0.5H2_0.5NH3
    :return:
    """
    logger.info(f"Using mechanism file: {mech}")
    logger.info(f"Using experiment results file: {exp_results}")
    logger.info(f"Using flame type: {flame_type}")

    exp_df = pd.read_csv(f"{input_dir}/{exp_results}")
    mech = (f"{mech_dir}/{mech}")
    # to get the filename of the mechanism
    mech_name = os.path.basename(mech).split(".")[0]

    # make a new dataframe of class objects
    classes = pd.DataFrame()

    # for each record in the dataframe we want to run it through a single condition
    # different types of flame can be run from the flame_type
    if flame_type == 'stagnation':
        classes["experiment_class"] = exp_df.apply(
            lambda row: StagnationFlame(
                {"O2": 0.21, "N2": 0.79},
                row["blend"],
                row["fuel"],
                row["phi"],
                row["T_in"],
                row["P"],
                row["T"],
                row["U"],
                260,
                mech_name,
            ),
            axis=1,
        )

    elif flame_type == 'freely_prop':
        classes["experiment_class"] = exp_df.apply(
            lambda row: FreelyPropFlame(
                {"O2": 0.21, "N2": 0.79},
                row["blend"],
                row["fuel"],
                row["phi"],
                row["T_in"],
                row["P"],
                260,
                mech_name,
            ),
            axis=1,
        )
    else:
        raise Exception('Invalid flame type')

    classes["cantera_gas"] = classes["experiment_class"].apply(
        lambda x: x.configure_gas()
    )
    classes["cantera_numerical_model"] = classes["experiment_class"].apply(
        lambda x: x.configure_flame()
    )
    tqdm.pandas(desc="Igniting Flames")
    classes["experiment_class"].progress_apply(lambda x: x.solve())

