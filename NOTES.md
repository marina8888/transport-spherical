****

<h4>Lets assume that the following process needs to take place, initially:  </h4>
1. Experimental data is taken as an input, and the mechanism(s) are run for this data <br>
2. An error is calculated between the numerical and experimental data<br>
3. A ROP and sensitivity graphs are run to identify the key transport, thermo etc. constants<br>

<h4>In dynamic mechanism development, lets assume the following process takes place: </h4>
1. A few key reactions/factors are identified and perturbed dynamically within user specifications<br>
2. The numerical result is saved and a new error factor is calculated<br>
3. The factor values and score are saved and input into an optimisation algorithm<br>
4. The factors are perturbed again, and until optimised values are reached<br>


<h3> TO DO: </h3>

<br>[x] Clean and create a csv input file format for experimental input conditions and data
<br>[x] Run numerical results based on an input file and get output numerical files in csv format for the output directory
<br>[x] Plot experimental and numerical data on one graph
<br>[x] Calculate the error between numerical and experimental data
<br>[x] Undertake a ROP and sensitivity analysis for the main reactions and paste to file, given a specific input condition
<br>[x] Undertake a thermal and transport file sensitivity analysis
<br>[ ] Integrate the above with UFlame
<br>[ ] Allow dynamic opimisation of constants, given experimental boundaries of data for each constant
<br>[ ] Calculate a SUE sensitivity given, given experimental boundaries of data for each constant
<br>[ ] Release as a pip python package

****
<h3> ROP AND SENSITIVITY ANALYSIS </h3>
The rate of production can only be undertaken on various species, so the following rules are made for the run_rops_sens:  <br><br>
1. If the user specifies a species like 'NO', and its searchable in the mechanism, we run the sensitivity and rop analysis as usual on the species. The ROP can only include reactions in thhe mechanism, but sensitivity should be able to include thermo and transport files <br>
2. If the user species 'lbv', we can only run sensitivity analysis on that. This should only be an option available for freely_prop <br>

****

<h3> ERROR EQUATION: </h3>

![img.png](resources/images/error_eq.png)

<h3> SENSITIVITY ANALYSIS </h3>
Sensitivity analysis is undertaken using the brute force method, where variables are perturbed. 
The simulated parameter is either laminar burning velocity (velocity at inlet) or species at outlet. 

![img.png](resources/images/brute_force.png)
