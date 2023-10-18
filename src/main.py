import pandas as pd
import matplotlib.pyplot as plt

from src.calculations.error_calculate import ErrorCalculator
from src.calculations.diffusion import TransportCalc

from src.plotter.numerical_vs_exp import plotter, plot_all
from src.plotter.numerical_only import plotter_domain, plotter_single, plotter_domain_sheet
from src.settings.filepaths import input_dir, mech_dir

from src.routes.call_flame import run_flame
from src.routes.rop_sens import run_rop_sens
from src.routes.call_domain import run_flame_domain




def main():
    # run flame simulation:
    # run_flame("okafor-mod.cti", "/stagnation/CH3_NH3_AH_sent_2ndprocessing/test.csv", flame_type="stagnation")
    # run_flame("creck.cti", "/stagnation/CH3_NH3_AH_sent_2ndprocessing/test2.csv", flame_type="stagnation")
    # run_flame("wang.cti", "/stagnation/CH3_NH3_AH_sent_2ndprocessing/test.csv", flame_type="stagnation")
    # run_flame("creck.cti", 'stagnation/CH3_NH3_AH_sent_2ndprocessing/20%_data_jp_symp.csv', flame_type='stagnation')
    # run_flame("arun.cti", 'stagnation/CH3_NH3_AH_sent_2ndprocessing/20%_data_jp_symp.csv', flame_type='stagnation')
    # plotting example:
    # plotter_single("1000grid/stagnation_CH4_NH3/20%", 'x_col', 'flame_speed', 'adiabatic flame temperature, '+ r"$\mathrm{T_{A}}$" + ' (K)', 'ammonia heat ratio, '+ r"$\mathrm{E_{NH3}}$", ['0.1MPa', '0.5MPa'])
    # plotter_single("1000grid/stagnation_CH4_NH3/60%", 'phi', 'NO',
    #                r"$\mathrm{X_{NO}}$" + ' (ppmv)',
    #                'equivalence ratio, ' + r"$\mathrm{\varphi}$", None, num_mulitplier=1000000) #(%, vol)
    # plotter("1000grid/stagnation_CH4_NH3_2nd_processing/20%", "/stagnation/CH3_NH3_AH_sent_2ndprocessing/20%_data_jp_symp.csv", "NO", 1, r"$\mathrm{X_{NO}}$" + ', (ppmv)', 1000000)
    # plot_all("stagnation/CH4_NH3_final", 'NO',1)

    # plotting domain example, mech only:
    # plotter_domain_sheet("../resources/output/numerical_domain/stagnation_CH4_NH3/0.2_strain_allE_0.9phi/ICFD_1bar_0.2_okafor-2017.csv", [0, 0.2, 0.4, 0.6, 0.8])

    # error calculation:
    # error_object = ErrorCalculator("1000grid/stagnation_CH4_NH3/60%", "stagnation/CH4_NH3/60%_data_reduced.csv", "stagnation")

    #sensitivity calculation:
    run_rop_sens("okafor-2017.yaml", "/stagnation/NH3_H2_N2/test.csv", flame_type="stagnation",
                  species='NO', type='sens_thermo')

if __name__ == "__main__":
    main()