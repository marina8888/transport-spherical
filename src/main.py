from src.calculations.error_calculate import ErrorCalculator
from src.plotter.numerical_vs_exp import plotter
from src.settings.filepaths import input_dir, mech_dir
from src.core.call_flame import run_flame
import cantera.cti2yaml as cli
# format of files should include Er for any non-phi variables
# Assume we are plotting by phi and segregating the blend plots
def main():
    run_flame("gri-liu.cti", "stagnation/CH4_NH3/20%_data.csv", flame_type="stagnation")
    run_flame("mei-2021.cti", "stagnation/CH4_NH3/20%_data.csv", flame_type="stagnation")
    run_flame("han-2020.cti", "stagnation/CH4_NH3/20%_data.csv", flame_type="stagnation")

    run_flame("okafor-2017.cti", "stagnation/CH4_NH3/60%_data_full.csv", flame_type="stagnation")
    run_flame("UCSD.cti", "stagnation/CH4_NH3/60%_data_full.csv", flame_type="stagnation")
    run_flame("han-2020.cti", "stagnation/CH4_NH3/60%_data_full.csv", flame_type="stagnation")
    run_flame("gri-liu.cti", "stagnation/CH4_NH3/60%_data_full.csv", flame_type="stagnation")
    run_flame("mei-2021.cti", "stagnation/CH4_NH3/60%_data_full.csv", flame_type="stagnation")

    # plotting example:
    # plotter("1000grid/stagnation_H2_NH3/0%", 'stagnation/NH3_H2/0%_data_full.csv', 'H2', 1, "X_H2 (vol, %)", 100)

    # error calculation:
    # error_object = ErrorCalculator("1000grid/stagnation_H2_NH3/0%", 'stagnation/NH3_H2/0%_data_full.csv', "stagnation")
if __name__ == "__main__":
    main()