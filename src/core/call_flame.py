from src.settings.filepaths import project_dir
import pandas as pd
from src.flames.stagnation_flame import StagnationFlame
from src.flames.freely_prop_flame import FreelyPropFlame
import os
from src.settings.logger import LogConfig
from tqdm import tqdm

logger = LogConfig.configure_logger(__name__)


def stagnation_main(mech, exp_results):
    logger.info(f"Using mechanism file: {mech}")
    logger.info(f"Using experiment results file: {exp_results}")

    exp_df = pd.read_csv(exp_results)
    # to get the filename of the mechanism
    mech_name = os.path.basename(mech)
    mech_name = mech_name.split(".")[0]

    # make a new dataframe of class objects
    classes = pd.DataFrame()

    # for each record in the dataframe we want to run it through a single condition
    # within the class ExperimentFlame, there are different functions that use Cantera core
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
    classes["cantera_gas"] = classes["experiment_class"].apply(
        lambda x: x.configure_gas()
    )
    classes["cantera_impinging_jet"] = classes["experiment_class"].apply(
        lambda x: x.configure_flame()
    )
    tqdm.pandas(desc="Igniting Flames")
    classes["experiment_class"].progress_apply(lambda x: x.solve())



def freely_prop_main(mech, exp_results):
    logger.info(f"Using mechanism file: {mech}")
    logger.info(f"Using experiment results file: {exp_results}")

    exp_df = pd.read_csv(exp_results)

    # to get the filename of the mechanism
    mech_name = os.path.basename(mech)
    mech_name = mech_name.split(".")[0]

    # make a new dataframe of class objects
    classes = pd.DataFrame()

    # for each record in the dataframe we want to run it through a single condition
    # within the class ExperimentFlame, there are different functions that use Cantera core
    classes["experiment_class"] = exp_df.apply(
        lambda row: StagnationFlame(
            {"O2": 0.21, "N2": 0.79},
            row["blend"],
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
    classes["cantera_gas"] = classes["experiment_class"].apply(
        lambda x: x.configure_gas()
    )
    classes["cantera_impinging_jet"] = classes["experiment_class"].apply(
        lambda x: x.configure_flame()
    )
    tqdm.pandas(desc="Igniting Flames")
    classes["experiment_class"].progress_apply(lambda x: x.solve())

