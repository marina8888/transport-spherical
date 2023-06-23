from src.settings.filepaths import mech_path, input_dir, output_dir
from core.call_flame import run_flame
def main():
    # run_flame(f"{mech_path}/okafor_2017.cti", f"{input_dir}/stagnation_CH4_NH3/20%_data.csv", flame_type = "stagnation")
    run_flame(f"{mech_path}/UCSD.cti", f"{input_dir}/freely_prop_H2_NH3/0.6H2_Han.csv", flame_type = "freely_prop")
    # run_flame(f"{mech_path}/wang.cti", f"{input_dir}/stagnation_CH4_NH3/20%_data.csv", flame_type = "stagnation")
    # run_flame(f"{mech_path}/shrestha.cti", f"{input_dir}/stagnation_CH4_NH3/20%_data.csv", flame_type = "stagnation")
    # run_flame(f"{mech_path}/mei.cti", f"{input_dir}/stagnation_CH4_NH3/20%_data.csv", flame_type = "stagnation")

if __name__ == "__main__":
    main()