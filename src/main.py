import pandas as pd
import matplotlib.pyplot as plt

from src.calculations.error_calculate import ErrorCalculator
from src.calculations.diffusion import TransportCalc

from src.plotter.numerical_vs_exp import plotter, plot_all
from src.plotter.numerical_only import plotter_domain, plotter_single, plotter_domain_sheet

from src.routes.call_flame import run_flame
from src.routes.rop_sens import run_rop_sens
from src.routes.call_domain import run_flame_domain


def main():
    # run flame simulations for full domain or end points only:
    run_flame("okafor-2017.cti", "stagnation/CH4_NH3_AH_sent_2ndprocessing/20%_fuelc.csv", flame_type="stagnation")
    # run_flame_domain("jiang.cti", "/stagnation/Andrea_paper/56%_N2_H2_bss.csv",  flame_type="stagnation")

    # data plotting example:
    # plotter_single("1000grid/stagnation_H2_NH3/30%", 'x_col', 'flame_speed', 'adiabatic flame temperature, '+ r"$\mathrm{T_{A}}$" + ' (K)', 'ammonia heat ratio, '+ r"$\mathrm{E_{NH3}}$", ['0.1MPa', '0.5MPa'])
    # plotter_single("1000grid/stagnation_H2_NH3/30%", 'phi', 'NO',
    #                 r"$\mathrm{X_{NO}}$" + ' (ppmv)',
    #                 'equivalence ratio, ' + r"$\mathrm{\varphi}$", None, num_mulitplier=1000000) #(%, vol)
    # plotter("1000grid/stagnation_CH4_NH3_2nd_processing/20%", "/stagnation/CH3_NH3_AH_sent_2ndprocessing/20%_data_jp_symp.csv", "NO", 1, r"$\mathrm{X_{NO}}$" + ', (ppmv)', 1000000)
    # plot_all("stagnation/CH4_NH3_final", 'NO',1)
    # plotter(numerical_folder="1000grid/stagnation_CH4_NH3_2nd_processing/20%", exp_results="stagnation/CH4_NH3_AH_sent_2ndprocessing/20%_data_jp_symp.csv", col = "NO", exp_multiplier = 1, y_label = r"$\mathrm{X_{NO}}$"+ ', (ppmv)',
    #             num_mulitplier=1000000)
    # plotter_domain_sheet("../resources/output/numerical_domain/stagnation_CH4_NH3/0.2_strain_allE_0.9phi/ICFD_1bar_0.2_okafor-2017.csv", [0, 0.2, 0.4, 0.6, 0.8])

    # error calculation:
    # error_object = ErrorCalculator("1000grid/stagnation_H2_NH3/0%", "/stagnation/NH3_H2/0%_data_full.csv", "stagnation")

    #sensitivity calculation:
    # run_rop_sens("wang.yaml", "/stagnation/CH4_NH3_AH_sent_2ndprocessing/test2.csv", flame_type="stagnation",
    #                   species='HCN', type='rop')
    # run_rop_sens("vargas.yaml", "/stagnation/Andrea_paper/30%_bss_0.7.csv", flame_type="freely_prop",
    #                   species='lbv', type='sens_adjoint')


if __name__ == "__main__":
    main()