from src.settings.filepaths import mech_path, input_dir, output_dir
from core.call_flame import stagnation_main, freely_prop_main
def main():
    stagnation_main(f"{mech_path}/okafor_2017.cti", f"{input_dir}/stagnation_CH4_NH3/20%_data.csv")

if __name__ == "__main__":
    main()