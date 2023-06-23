
from core.call_flame import run_flame
from core.error_calculate import ErrorCalculator
from plotter.numerical_vs_exp import plotter
def main():
    # run_flame(f"{mech_path}/okafor-2017.cti", f"{input_dir}/stagnation_CH4_NH3/20%_data.csv", flame_type = "stagnation")
    # run_flame(f"{mech_path}/okafor-2017.cti", f"{input_dir}/freely_prop_0.5H2_0.5NH3/Lhulier_data_0.6.csv", flame_type = "freely_prop_0.5H2_0.5NH3")
    # run_flame(f"{mech_path}/shrestha.cti", f"{input_dir}/freely_prop_0.5H2_0.5NH3/Lhulier_data_0.6.csv", flame_type="freely_prop_0.5H2_0.5NH3")
    # run_flame(f"wang.cti", "/freely_prop_0.5H2_0.5NH3/Lhulier_data_0.6.csv", flame_type="freely_prop_0.5H2_0.5NH3")
    # run_flame(f"{mech_path}/UCSD.cti", f"{input_dir}/freely_prop_0.5H2_0.5NH3/0.6H2_Han.csv", flame_type="stagnation")
    # run_flame(f"{mech_path}/wang.cti", f"{input_dir}/stagnation_CH4_NH3/20%_data.csv", flame_type = "stagnation")
    # run_flame(f"{mech_path}/shrestha.cti", f"{input_dir}/stagnation_CH4_NH3/20%_data.csv", flame_type = "stagnation")
    # run_flame(f"{mech_path}/mei.cti", f"{input_dir}/stagnation_CH4_NH3/20%_data.csv", flame_type = "stagnation")
    # error_object = ErrorCalculator(exp_results: , name_of_numerical_folder: str, flame_type: str)
    plotter("freely_prop_0.5H2_0.5NH3", 'freely_prop_H2_NH3/Lhulier_data_0.6.csv', 'flame_speed', 1, "laminar burning velocity (cm/s)", 100)



if __name__ == "__main__":
    main()