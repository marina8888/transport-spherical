# transport-spherical
Code to dynamically modify reaction constants for combustion kinetics mechanisms based on results of LBV, unstretched LBV and stagnation flame

<h3> TO DO: </h3>

<br>[ ] Clean and create a csv input file format for experimental input conditions and data
<br>[ ] Run numerical results based on an input file and get output numerical files in csv format for the output directory
<br>[ ] Plot experimental and numerical data on one graph
<br>[ ] Calculate the error between numerical and experimental data
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
