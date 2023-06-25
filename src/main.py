from src.calculations.error_calculate import ErrorCalculator
from src.plotter.numerical_vs_exp import plotter
from src.settings.filepaths import input_dir, mech_dir
from src.core.call_flame import run_flame

# format of files should include Er for any non-phi variables
# Assume we are plotting by phi and segregating the blend plots
def main():
    # run_flame("UCSD.cti", "freely_prop_H2_NH3/Lhulier_data_0.6.csv", flame_type = "freely_prop")
    # run_flame(f"{mech_path}/okafor-2017.cti", f"{input_dir}/freely_prop_0.5H2_0.5NH3/Lhulier_data_0.6.csv", flame_type = "freely_prop_0.5H2_0.5NH3")
    # run_flame(f"{mech_path}/shrestha.cti", f"{input_dir}/freely_prop_0.5H2_0.5NH3/Lhulier_data_0.6.csv", flame_type="freely_prop_0.5H2_0.5NH3")
    # run_flame(f"wang.cti", "/freely_prop_0.5H2_0.5NH3/Lhulier_data_0.6.csv", flame_type="freely_prop_0.5H2_0.5NH3")
    # run_flame(f"{mech_path}/UCSD.cti", f"{input_dir}/freely_prop_0.5H2_0.5NH3/0.6H2_Han.csv", flame_type="stagnation")
    # run_flame("wang.cti","CH4_NH3/20%_data.csv", flame_type = "stagnation")
    # run_flame("UCSD.cti", "CH4_NH3/20%_data.csv", flame_type = "stagnation")
    # run_flame("okafor-2017.cti", "CH4_NH3/20%_data.csv", flame_type = "stagnation")
    # run_flame(f"{mech_path}/mei.cti", f"{input_dir}/CH4_NH3/20%_data.csv", flame_type = "stagnation")

    # plotting example:
    plotter("1000grid/stagnation_H2_NH3/20%", 'stagnation/NH3_H2/20%_data_full.csv', 'H2O', 1, "X_H2O (ppmv)", 1000000)

    # error calculation:
    # error_object = ErrorCalculator("1000grid/stagnation_H2_NH3/30%", 'stagnation/NH3_H2/30%_data_full.csv', "stagnation")
if __name__ == "__main__":
    main()