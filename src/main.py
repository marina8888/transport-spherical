import pandas as pd
import matplotlib.pyplot as plt
from src.calculations.error_calculate import ErrorCalculator
from src.plotter.numerical_vs_exp import plotter
from src.plotter.numerical_only import plotter_domain, plotter_single, plotter_domain_sheet
from src.settings.filepaths import input_dir, mech_dir
from src.core.call_flame import run_flame
from src.core.call_rop import run_rop_sens
from src.core.call_domain import run_flame_domain
from src.calculations.diffusion import TransportCalc
import cantera.cti2yaml as cli

def main():
    run_flame_domain("okafor-2017.cti", '/stagnation/CH4_NH3/icfd_2023_reduced.csv', flame_type='stagnation')
    # run_flame("okafor-2017.cti", "freely_prop/CH4_NH3/icfd_lbv_test.csv", flame_type="freely_prop")

    # plotting example:
    # plotter_single("1000grid/stagnation_CH4_NH3/0.2_strain_allE_0.9phi", 'x_col', 'flame_speed', 'adiabatic flame temperature, '+ r"$\mathrm{T_{A}}$" + ' (K)', 'ammonia heat ratio, '+ r"$\mathrm{E_{NH3}}$", ['0.1MPa', '0.5MPa'])


    # plotting domain example, mech only:
    # plotter_domain_sheet("stagnation_CH4_NH3/0.2_strain_allE_0.9phi/ICFD_5bar_0.2_okafor-2017.csv", [0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # error calculation:
    # error_object = ErrorCalculator("1000grid/stagnation_H2_NH3/10%", 'stagnation/NH3_H2/10%_data_full.csv', "stagnation")

    #sensitivity calculation:
    # run_rop_sens("arun.cti", "stagnation/CH4_NH3/20%_data_jp_symp.csv", flame_type="freely_prop", species='HCN')

if __name__ == "__main__":
    main()