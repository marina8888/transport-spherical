import json
from pathlib import Path


def load_config(file_path='../resources/input/config/config.json'):
    with open(file_path, 'r') as file:
        config_data = json.load(file)
    return config_data

def set_global_variables():
    print("setting global variables...")
    global SLOPE, CURVE, PRUNE, MAX_GRID, STAGNATION_FLAME_WIDTH, FREELY_PROP_FLAME_WIDTH
    global SENS_SPECIES_LOC, ROP_LOC_FROM, ROP_LOC_TO, SENS_PLOT_FILTER, ROP_PLOT_FILTER, SENS_PERTURBATION
    global INPUT_DIR_NUMERICAL, INPUT_DIR_MECH, OUTPUT_DIR_NUMERICAL, OUTPUT_DIR_DOMAIN, OUTPUT_DIR_ERROR, OUTPUT_DIR_ROP, OUTPUT_DIR_SENS, OUTPUT_DIR_THERMO
    global GRAPHS_DIR, GRAPHS_DIR_ROP, GRAPHS_DIR_SENS, GRAPHS_DIR_THERMO
    config_data = load_config()

    # simulation run settings:
    SLOPE = config_data['grid']['slope']
    CURVE = config_data['grid']['curve']
    PRUNE = config_data['grid']['prune']
    MAX_GRID = config_data['grid']['max_grid']
    STAGNATION_FLAME_WIDTH = config_data['grid']['stagnation_flame_width']
    FREELY_PROP_FLAME_WIDTH = config_data['grid']['freely_prop_flame_width']

    # species and rop analysis configuration settings:
    SENS_SPECIES_LOC = config_data['sens_rop']['sensitivity_species_loc']
    ROP_LOC_FROM = config_data['sens_rop']['rop_loc_from']
    ROP_LOC_TO = config_data['sens_rop']['rop_loc_to']
    SENS_PLOT_FILTER = config_data['sens_rop']['sensitivity_plot_filter']
    ROP_PLOT_FILTER = config_data['sens_rop']['rop_plot_filter']
    SENS_PERTURBATION = config_data['sens_rop']['sensitivity_perturbation']

    # directories for finding files and saving them:
    src_dir = Path(__file__).parent.parent.as_posix()
    PROJECT_DIR = Path(src_dir).parent.as_posix()

    INPUT_DIR_NUMERICAL = f"{PROJECT_DIR}{config_data['locations']['input_dir_numerical']}"
    INPUT_DIR_MECH = f"{PROJECT_DIR}{config_data['locations']['input_dir_mech']}"

    OUTPUT_DIR_NUMERICAL = f"{PROJECT_DIR}{config_data['locations']['output_dir_numerical']}"
    OUTPUT_DIR_DOMAIN = f"{PROJECT_DIR}{config_data['locations']['output_dir_domain']}"
    OUTPUT_DIR_NUMERICAL = f"{PROJECT_DIR}{config_data['locations']['output_dir_numerical']}"
    OUTPUT_DIR_ERROR = f"{PROJECT_DIR}{config_data['locations']['output_dir_error']}"
    OUTPUT_DIR_ROP = f"{PROJECT_DIR}{config_data['locations']['output_dir_rop']}"
    OUTPUT_DIR_SENS = f"{PROJECT_DIR}{config_data['locations']['output_dir_sens']}"
    OUTPUT_DIR_THERMO = f"{PROJECT_DIR}{config_data['locations']['output_dir_thermo']}"

    GRAPHS_DIR = f"{PROJECT_DIR}{config_data['locations']['graphs_dir']}"
    GRAPHS_DIR_ROP = f"{PROJECT_DIR}{config_data['locations']['graphs_dir_rop']}"
    GRAPHS_DIR_SENS = f"{PROJECT_DIR}{config_data['locations']['graphs_dir_sens']}"
    GRAPHS_DIR_THERMO = f"{PROJECT_DIR}{config_data['locations']['graphs_dir_thermo']}"

set_global_variables()

