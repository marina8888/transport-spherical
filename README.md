# transport-spherical

Code to dynamically modify reaction constants for combustion kinetics mechanisms based on results of LBV, unstretched LBV and stagnation flame

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
|   ├── output
|   ├── mech
|   ├──templates
|-- src
|   ├── core
|   |   ├── call a flame and print numerical results to file
|   |   ├── call a flame and get ROPs and sensitivities
|   ├── calculations
|   |   ├── convert between x and y error
|   |   ├── find the y values of interest
|   |   ├── calculate error between experimental and numerical results sheets
|   |   ├── call the optimisation algorithm
|   ├── flames
|   |   ├── stagnation flame
|   |   ├── freely-propagating
|   ├── settings
|   |   ├── define project file paths
|   |   ├── define project logging
|   ├── utils (processing tools)
  └── README.md
</code>
</pre>
