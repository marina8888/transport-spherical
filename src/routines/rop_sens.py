import pandas as pd
import os
from tqdm import tqdm

from src.flames.stagnation_flame import StagnationFlame
from src.flames.freely_prop_flame import FreelyPropFlame

from src.settings.logger import LogConfig
import src.settings.config_loader as config


# this file calls various flame types to get all rop

logger = LogConfig.configure_logger(__name__)

def  run_rop_sens(mech:str, exp_results:str, flame_type:str, species: str, type = 'sens_adjoint'):
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

    exp_df = pd.read_csv(f"{config.INPUT_DIR_NUMERICAL}/{exp_results}")
    mech = (f"{config.INPUT_DIR_MECH}/{mech}")
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
    match type:
        case 'sens_adjoint':
            logger.info(f"Calculating sensitivity for {species} wrt reaction A factor coeffs using solver adjoint")
            classes["experiment_class"].apply(lambda x: x.get_sens_adjoint())

        case 'sens_brute_force':
            logger.info(f"Calculating sensitivity for {species} wrt reaction A factor coeffs using brute force")
            classes["experiment_class"].apply(lambda x: x.get_sens_brute_force())

        case 'sens_thermo':
            logger.info(f"Calculating Sensitivity for {species} wrt enthalpy and entropy")
            classes["experiment_class"].apply(lambda x: x.get_sens_thermo())






        # type of ROP analysis:
        case 'rop_all':
            logger.info(f"Calculating ROP for all species")
            classes["experiment_class"].apply(lambda x: x.get_rop_all())

        case 'rop':
            logger.info(f"Calculating ROP for {species}")
            classes["experiment_class"].apply(lambda x: x.get_rop())

        case 'rop_distance':
            logger.info(f"Calculating ROP for {species}")
            classes["experiment_class"].apply(lambda x: x.get_rop_distance())

        case _:
            print('Argument not recognised: not doing any ROP and sensitivity')

