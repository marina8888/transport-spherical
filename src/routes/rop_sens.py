import pandas as pd
from src.settings.filepaths import input_dir, output_dir, mech_dir
from src.flames.stagnation_flame import StagnationFlame
from src.flames.freely_prop_flame import FreelyPropFlame
import os
from src.settings.logger import LogConfig
from tqdm import tqdm

# this file calls various flame types to get all rop

logger = LogConfig.configure_logger(__name__)

def run_rop_sens(mech:str, exp_results:str, flame_type:str, species: str, type = 'sens_adjoint'):
    """
    @param mech: input mechanism
    @param exp_results: csv file of exp data
    @param flame_type: 'stagnation' or 'freely_prop'
    @param conditions: list of integer numbers corresponding to csv file lines
    @param species: species to calculate ROP for
    @param type: 'rop', 'rop_all', 'sens_adjoint', 'sens_brute_force', 'sens_thermo', 'sens_trans'
    @return:
    """
    logger.info(f"Using mechanism file: {mech}")
    logger.info(f"Using experiment results file: {exp_results}")
    logger.info(f"Using flame type: {flame_type}")
    logger.info(f"Calculating ROP for species: {species}")

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
                row['oxidizer'],
                row["blend"],
                row["fuel"],
                row["phi"],
                row["T_in"],
                row["P"],
                row["T"],
                row["U"],
                mech_name,
                species
            ),
            axis=1,
        )

    elif flame_type == 'freely_prop':
        classes["experiment_class"] = exp_df.apply(
            lambda row: FreelyPropFlame(
                row['oxidizer'],
                row["blend"],
                row["fuel"],
                row["phi"],
                row["T_in"],
                row["P"],
                mech_name,
                species,
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


    # type of sensitivity analysis:

    if type == 'sens_adjoint':
        classes["experiment_class"].apply(lambda x: x.get_sens_adjoint())

    elif type == 'sens_brute_force':
        classes["experiment_class"].apply(lambda x: x.get_sens_brute_force())

    elif type == 'sens_thermo':
        classes["experiment_class"].apply(lambda x: x.get_sens_thermo())

    elif type == 'sens_trans':
        classes["experiment_class"].apply(lambda x: x.get_sens_trans())

    else:
        print('Sensitivity analysis input argument not recognised - not running a sensitivity analysis')


    # type of ROP analysis:

    if type == 'rop_all':
        classes["experiment_class"].apply(lambda x: x.get_rop_all())

    elif type == 'rop':
        classes["experiment_class"].apply(lambda x: x.get_rop())

    else:
        print('ROP analysis input argument not recognised - not running an ROP analysis')
