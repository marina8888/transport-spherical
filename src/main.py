from src.calculations.error_calculate import ErrorCalculator
from src.plotter.numerical_vs_exp import plotter
from src.settings.filepaths import input_dir, mech_dir
from src.core.call_flame import run_flame
from src.calculations.diffusion import TransportCalc
import cantera.cti2yaml as cli

def main():
    # cli.main('--input=resources/mech-chemkin/mei-2021/mech_chemkin/mech.inp',
    #                     '--thermo=.resources/mech-chemkin/mei-2021/mech_chemkin/thermo.dat',
    #                     '--transport=resources/mech-chemkin/mei-2021/mech_chemkin/trans.inp', '--permissive')

    # run_flame("gri-liu.cti", "stagnation/CH4_NH3/20%_data.csv", flame_type="stagnation")
    # run_flame("gri.cti", "stagnation/CH4_NH3/20%_data.csv", flame_type="stagnation")
    run_flame("mei2021.cti", "stagnation/CH4_NH3/20%_data.csv", flame_type="stagnation")
    #
    #
    # run_flame("okafor-2017.cti", "stagnation/CH4_NH3/60%_data_full.csv", flame_type="stagnation")
    # run_flame("UCSD.cti", "stagnation/CH4_NH3/60%_data_full.csv", flame_type="stagnation")
    # run_flame("mei2021.cti", "stagnation/CH4_NH3/60%_data_full.csv", flame_type="stagnation")
    # run_flame("gri-liu.cti", "stagnat
    #
    #
    # ion/CH4_NH3/60%_data_full.csv", flame_type="stagnation")
    # run_flame("mei2021.cti", "stagnation/CH4_NH3/60%_data_full.csv", flame_type="stagnation")

    # plotting example:
    # plotter("1000grid/stagnation_H2_NH3/0%", 'stagnation/NH3_H2/0%_data_full.csv', 'H2', 1, "X_H2 (vol, %)", 100)

    # error calculation:
    # error_object = ErrorCalculator("1000grid/stagnation_H2_NH3/10%", 'stagnation/NH3_H2/10%_data_full.csv', "stagnation")
if __name__ == "__main__":
    main()