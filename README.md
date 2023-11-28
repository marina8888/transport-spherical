# transport-spherical
****
<h3> HOW TO RUN </h3>
1. Update the file in resources/inputs/config to the correct simulation parameters (for example grid options) <br>
2. Based on resources/templates create an input excel file of all the conditions to run. The 'blend' column will be your file name defintion. If this changes half way in the input sheet, a new results sheet to print results will be created. Feel free to use this to split up runs in one input file. <br>
3. in the main.py, main() function, paste one of the following functions:<br>
<code>

      # run flame simulation (two types of flame) and save the outputs:
    run_flame("creck.yaml", "/stagnation/Andrea_paper/30%_bss.csv", flame_type="stagnation")
    run_flame("creck.yaml", "/stagnation/CH3_NH3_AH_sent_2ndprocessing/test2.csv", flame_type="freely_prop")


    # plotting examples:
    plotter_single("1000grid/stagnation_H2_NH3/30%", 'x_col', 'flame_speed', 'adiabatic flame temperature, '+ r"$\mathrm{T_{A}}$" + ' (K)', 'ammonia heat ratio, '+ r"$\mathrm{E_{NH3}}$", ['0.1MPa', '0.5MPa'])
    plotter_single("1000grid/stagnation_H2_NH3/30%", 'phi', 'NO', r"$\mathrm{X_{NO}}$" + ' (ppmv)', 'equivalence ratio, ' + r"$\mathrm{\varphi}$", None, num_mulitplier=1000000) #(%, vol)
    plotter("1000grid/stagnation_CH4_NH3_2nd_processing/20%", "/stagnation/CH3_NH3_AH_sent_2ndprocessing/20%_data_jp_symp.csv", "NO", 1, r"$\mathrm{X_{NO}}$" + ', (ppmv)', 1000000)
    plot_all("stagnation/CH4_NH3_final", 'NO',1)
    plotter(numerical_folder="1000grid/stagnation_CH4_NH3_2nd_processing/20%", exp_results="stagnation/CH4_NH3_AH_sent_2ndprocessing/20%_data_jp_symp.csv", col = "NO", exp_multiplier = 1, y_label = r"$\mathrm{X_{NO}}$"+ ', (ppmv)', num_mulitplier=1000000)

    # plotting domain example, mech only:
    plotter_domain_sheet("../resources/output/numerical_domain/stagnation_CH4_NH3/0.2_strain_allE_0.9phi/ICFD_1bar_0.2_okafor-2017.csv", [0, 0.2, 0.4, 0.6, 0.8])

    # error calculation:
    error_object = ErrorCalculator("1000grid/stagnation_H2_NH3/0%", "/stagnation/NH3_H2/0%_data_full.csv", "stagnation")

    #sensitivity calculation (top is ROP analysis, second is equillibrium). 
    run_rop_sens("wang.yaml", "/stagnation/CH4_NH3_AH_sent_2ndprocessing/test2.csv", flame_type="stagnation", species='HCN', type='rop')
    run_rop_sens("jiang.yaml", "/stagnation/NH3_H2/test.csv", flame_type="stagnation", species='N2O', type='sens_thermo')

    # List of available sens, rop analysis: 'sens_adjoint', 'sens_brute_force', 'sens_thermo', 'rop_all', 'rop', 'rop_distance'
</code>

****
****
<h3> INPUT FLAME TYPES CURRENTLY AVAILABLE AS INPUTS IN THE CORE FUNCTIONS </h3>
'stagnation' for a ImpingingJet flame <br>
'freely_prop' for a FreelyPropagating flame <br>

****


<h3> PROJECT STRUCTURE </h3>
<pre>
<code>
root
|-- resources
|   ├── input
|   |   ├── config (adjust file settings here)
|   |   ├── flame types (input excel sheet with conditions here)  
|   ├── output
|   ├── mech
|   ├──templates (templates for generating excel sheet)
|-- src
|   ├── routes
|   |   ├── call_flame (get output emissions or LBV only)
|   |   ├── flame_domain (get values across domain)
|   |   ├── rop_sens (get rop, sensitivity analysis)
|   ├── calculations
|   |   ├── convert between x and y error
|   |   ├── find the y values of interest
|   |   ├── calculate error between experimental and numerical results sheets
|   |   ├── call the optimisation algorithm
|   |   ├── rop and sensitivity analysis files
|   |   ├── transport files
|   ├── flames (easily scalable to include other flame types - add new flame files here)
|   |   ├── stagnation flame
|   |   ├── freely-propagating
|   ├── settings
|   |   ├── logger file class
|   |   ├── configuration file
|   ├── utils (processing tools)
  └── README.md
</code>
</pre>
