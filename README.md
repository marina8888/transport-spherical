# transport-spherical
Code to dynamically modify reaction constants for combustion kinetics mechanisms based on results of LBV, unstretched LBV and stagnation flame

<h3> TO DO: </h3>
- [ ] Clean and create a csv input file format for experimental input conditions and data
- [ ] Run numerical results based on an input file and get output numerical files in csv format for the output directory
- [ ] Plot experimental and numerical data on one graph
- [ ] Calculate the error between numerical and experimental data
- [ ] Undertake a ROP and sensitivity analysis for the main reactions and paste to file, given a specific input condition
- [ ] Understake a thermal and transport file sensitivity analysis
****

<h3> FUTURE WORK: </h3>
- [ ] Integrate the above with UFlame
- [ ] Allow dynamic opimisation of constants, given experimental boundaries of data for each constant
- [ ] Calculate a SUE sensitivity given, given experimental boundaries of data for each constant
- [ ] Release as a pip python package
****


<h3> PROJECT STRUCTURE </h3>
├── root
│   ├── resources
│   │   ├── input
│   │   ├── output
│   │   ├── mechs
│   │   ├── templates
│   ├── src
│   │   ├── core
│   │   ├── flames (stagnation and freely propagating)
│   │   ├── settings (for project structure)
│   │   ├── utils (processing tools)
│   └── README.md