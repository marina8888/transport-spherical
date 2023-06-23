# transport-spherical
Code to dynamically modify reaction constants for combustion kinetics mechanisms based on results of LBV, unstretched LBV and stagnation flame

<h4>Lets assume that the following process needs to take place, initially:  </h4><br>
1. Experimental data is taken as an input, and the mechanism(s) are run for this data <br>
2. An error is calculated between the numerical and experimental data<br>
3. A ROP and sensitivity graphs are run to identify the key transport, thermo etc. constants<br>

<h4>In dynamic mechanism development, lets assume the following process takes place: </h4><br>
1. A few key reactions/factors are identified and perturbed dynamically within user specifications<br>
2. The numerical result is saved and a new error factor is calculated<br>
3. The factor values and score are saved and input into an optimisation algorithm<br>
3. The factors are perturbed again, and until optimised values are reached<br>


<h3> TO DO: </h3>

<br>[x] Clean and create a csv input file format for experimental input conditions and data
<br>[x] Run numerical results based on an input file and get output numerical files in csv format for the output directory
<br>[x] Plot experimental and numerical data on one graph
<br>[x] Calculate the error between numerical and experimental data
<br>[ ] Undertake a ROP and sensitivity analysis for the main reactions and paste to file, given a specific input condition
<br>[ ] Understake a thermal and transport file sensitivity analysis

****

<h3> FUTURE WORK: </h3>
<br>[ ] Integrate the above with UFlame
<br>[ ] Allow dynamic opimisation of constants, given experimental boundaries of data for each constant
<br>[ ] Calculate a SUE sensitivity given, given experimental boundaries of data for each constant
<br>[ ] Release as a pip python package
****


<h3> PROJECT STRUCTURE </h3>
<pre>
<code>
root
|-- resources
|   ├── input
|   ├── output
|   ├── mech
|   ├──templates
|-- src
|   ├── core
|   ├── flames (stagnation and freely propagating)
|   ├── settings (for project structure)
|   ├── utils (processing tools)
  └── README.md
</code>
</pre>
