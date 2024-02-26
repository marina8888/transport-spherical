import pandas as pd
import matplotlib.pyplot as plt

from src.calculations.error_calculate import ErrorCalculator
from src.calculations.diffusion import TransportCalc
from src.calculations.uncert_band_nasa import calculate

from src.plotter.numerical_vs_exp import plotter, plot_all
from src.plotter.numerical_only import plotter_domain, plotter_single, plotter_domain_sheet, plotter_single_input

from src.routines.call_flame import run_flame
from src.routines.rop_sens import run_rop_sens
from src.routines.call_domain import run_flame_domain



def main():
    # run flame simulations for full domain or end points only:
    # run_flame("gotama.yaml", "stagnation/NH3_H2/30%_lf_p_test.csv", flame_type="freely_prop")
    # run_flame("gotama.yaml", "stagnation/NH3_H2/30%_lf_p_test.csv", flame_type="stagnation")
    # run_flame("wang.yaml", "/stagnation/CH4_NH3_AH_sent_2ndprocessing/30%_fuelc.csv", flame_type="stagnation")
    # run_flame("wang.yaml", "/stagnation/CH4_NH3_AH_sent_2ndprocessing/40%_fuelc.csv", flame_type="stagnation")
    # run_flame("wang.yaml", "/stagnation/CH4_NH3_AH_sent_2ndprocessing/60%_fuelc.csv", flame_type="stagnation")

    # run_flame_domain("jiang.cti", "/stagnation/Andrea_paper/56%_N2_H2_bss.csv",  flame_type="stagnation")

    # effects of external variables:
    # plotter_single_input("stagnation/CH4_NH3_AH_sent_2ndprocessing/co2_2.csv", 'Qd Ar/CO2 upper', 'X_NO', 'dilution gas flowrate, (SLM)', 'absolute change in X, (ppm)', num_mulitplier=1)
    # plotter_single("1000grid/stagnation_H2_NH3/30%", 'phi', 'NO',
    #                 r"$\mathrm{X_{NO}}$" + ' (ppmv)',
    #                 'equivalence ratio, ' + r"$\mathrm{\varphi}$", None, num_mulitplier=1000000) #(%, vol)
    # plotter("1000grid/stagnation_H2_NH3/0%","stagnation/NH3_H2/0%_data.csv",
    #         "NO", 1, r"$\mathrm{X_{NO}}$" + ', (ppmv)', 1000000, title='E = 100%')
    # plotter("1000grid/stagnation_CH4_NH3_1st_processing/10%", "stagnation/CH4_NH3_AH_sent_2ndprocessing/10%_fuelc.csv", "CH4", 1, r"$\mathrm{X_{CH4}}$" + ', (ppmv)', 1000000, title = 'E = 10%')
    # plotter("1000grid/stagnation_CH4_NH3_2nd_processing/10%", "stagnation/CH4_NH3_AH_sent_2ndprocessing/10%_fuelc.csv", "CO", 1, r"$\mathrm{X_{CO}}$" + ', (ppmv)', 1000000)
    # plot_all("stagnation/CH4_NH3_final", 'NO',1)
    # plotter(numerical_folder="1000grid/stagnation_CH4_NH3_2nd_processing/20%", exp_results="stagnation/CH4_NH3_AH_sent_2ndprocessing/20%_data_jp_symp.csv", col = "NO", exp_multiplier = 1, y_label = r"$\mathrm{X_{NO}}$"+ ', (ppmv)',
    #             num_mulitplier=1000000)
    # plotter_domain_sheet("../resources/output/numerical_domain/stagnation_CH4_NH3/0.2_strain_allE_0.9phi/ICFD_1bar_0.2_okafor-2017.csv", [0, 0.2, 0.4, 0.6, 0.8])

    # error calculation:
    # error_object = ErrorCalculator("1000grid/stagnation_CH4_NH3_1st_processing/60%", "/stagnation/CH4_NH3_final_1stprocessing/60%_data_reduced.csv", "stagnation")

    #sensitivity calculation:
    # run_rop_sens("okafor-2017.yaml", "/stagnation/CH4_NH3_AH_sent_2ndprocessing/test2.csv", flame_type="stagnation",
    #                     species='NO', type='sens_thermo')
    run_rop_sens("okafor-2017.yaml", "/stagnation/CH4_NH3_AH_sent_2ndprocessing/test2.csv", flame_type="stagnation",
                         species='NH3', type='sens_adjoint')
    # run_rop_sens("okafor-2017.yaml", "/stagnation/CH4_NH3_AH_sent_2ndprocessing/test2.csv", flame_type="freely_prop",
    #                     species='lbv', type='sens_adjoint')
    # run_rop_sens("okafor-2017.yaml", "/stagnation/CH4_NH3_AH_sent_2ndprocessing/test2.csv", flame_type="stagnation",
    #                    species='NO2', type='sens_thermo')
    # run_rop_sens("okafor-2017.yaml", "/stagnation/CH4_NH3_AH_sent_2ndprocessing/test2.csv", flame_type="stagnation",
    #                     species='NH3', type='sens_thermo')
    # run_rop_sens("okafor-2017.yaml", "/stagnation/CH4_NH3_AH_sent_2ndprocessing/test2.csv", flame_type="stagnation",
    #                     species='NO', type='sens_thermo')
    # run_rop_sens("okafor-2017.yaml", "/stagnation/CH4_NH3_AH_sent_2ndprocessing/test2.csv", flame_type="stagnation",
    #                    species='H2', type='sens_thermo')

    # calculate(["okafor-2017.yaml", "mathieu.yaml", "gri.yaml", "creck.yaml", "arun.yaml", "jiang.yaml", "UCSD.yaml", "wang.yaml"])



if __name__ == "__main__":
    main()