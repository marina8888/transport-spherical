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

    # run_flame("wang.cti", '/stagnation/CH4_NH3/10%_data_rich.csv', flame_type='stagnation')
    # run_flame("zhang.cti", '/stagnation/CH4_NH3/10%_data_reduced.csv', flame_type='stagnation')
    # run_flame("okafor-2017.cti", '/stagnation/CH4_NH3/40%_data_reduced.csv', flame_type='stagnation')
    # run_flame("gri.cti", '/stagnation/CH4_NH3/40%_data_reduced.csv', flame_type='stagnation')
    # run_flame("UCSD.cti", '/stagnation/CH4_NH3/40%_data_reduced.csv', flame_type='stagnation')

    # run_flame("okafor-2017.cti", '/stagnation/CH4_NH3/60%_data_reduced.csv', flame_type='stagnation')
    # run_flame("gri.cti", '/stagnation/CH4_NH3/60%_data_reduced.csv', flame_type='stagnation')
    # run_flame("UCSD.cti", '/stagnation/CH4_NH3/60%_data_reduced.csv', flame_type='stagnation')
    # run_flame("okafor-2017.cti", "freely_prop/CH4_NH3/icfd_lbv_test.csv", flame_type="freely_prop")
    # run_flame("creck.cti", '/stagnation/CH4_NH3/60%_data_reduced.csv', flame_type='stagnation')
    # run_flame("creck.cti", '/stagnation/CH4_NH3/40%_data_reduced.csv', flame_type='stagnation')
    # run_flame("creck.cti", '/stagnation/CH4_NH3/30%_data_reduced.csv', flame_type='stagnation')
    run_flame("creck.cti", '/stagnation/CH4_NH3/20%_data_reduced.csv', flame_type='stagnation')
    run_flame("creck.cti", '/stagnation/CH4_NH3/10%_data_reduced.csv', flame_type='stagnation')

    # plotting example:
    # plotter_single("1000grid/stagnation_CH4_NH3/20%", 'x_col', 'flame_speed', 'adiabatic flame temperature, '+ r"$\mathrm{T_{A}}$" + ' (K)', 'ammonia heat ratio, '+ r"$\mathrm{E_{NH3}}$", ['0.1MPa', '0.5MPa'])
    # plotter_single("1000grid/stagnation_H2_NH3/20%", 'phi', 'NO',
    #                r"$\mathrm{X_{NO}}$" + ' (ppmv)',
    #                'ammonia heat ratio, ' + r"$\mathrm{E_{NH3}}$", None, num_mulitplier=1000000) (%, vol)
    # plotter("1000grid/stagnation_H2_NH3/30%", "stagnation/NH3_H2/30%_data_full.csv", "NO2", 1, r"$\mathrm{X_{NO2}}$" + ', (ppmv)', 1000000)
    # plotting domain example, mech only:
    # plotter_domain_sheet("../resources/output/numerical_domain/stagnation_CH4_NH3/0.2_strain_allE_0.9phi/ICFD_1bar_0.2_okafor-2017.csv", [0, 0.2, 0.4, 0.6, 0.8])

    # error calculation:
    # error_object = ErrorCalculator("1000grid/stagnation_H2_NH3/30%", 'stagnation/NH3_H2/30%_data_full.csv', "stagnation")

    #sensitivity calculation:
    # run_rop_sens("arun.cti", "stagnation/CH4_NH3/20%_data_jp_symp.csv", flame_type="freely_prop", species='HCN')

if __name__ == "__main__":
    main()