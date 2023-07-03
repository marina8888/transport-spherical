<h3> 01/07/2023 </h3> <br>
- Which parameters do we want to perturb? For example, all NASA polynomial coefficients directly or just the calculated enthalpy, specific heat etc values? <br>
- Do we want them on seperate graphs (preferable) - so for example the thermo sensitivity will be a list of species along the y axis (based on modification to NASA polynomial) and sensitivities along the x axis. <br>
- Meanwhile, the transport graphs will look like...? Each species has a few transport parameters - just do all independently (so 5 x number of species of perturbations available on y axis list)?<br>
- Which databases do we use to define our uncertainty boundaries? What kind of approach do we use - i.e spline uncertainty fitting, monte carlo sampling, just brute force across all polynomial parameters? <br>
- Do we want to perturb all the arrhenius parameters (and is it valid to do so) or just A? <br>
- Are we happy to try to introduce this method for stretched flame speed calculations (via. UFLame) - would this actually be helpful? <br>

<h3> 03/07/2023 - Okafor-sensei meeting</h3> <br>
- Ideally we want to try to implement the solver adjoint method instead of brute force <br>
- We shouldn't do it on stetched flames for now - (but I still need to reply to Brian Maxwell with an update) <br>
- We want to perturb the reverse reaction rate. This is dependent on enthalpy and entropy. So actually we want to perturb the enthalpy and entropy out of the thermo file, and not the NASA polynomial coefficients.  <br> 
- Instead of calculating the entropy, we might need to go into the calculation and add a multiplicaion factor. 
- For the transport properties - we only want to focus on 1-2 transport properties, check on NIST database which properties vary the most.<br> 
- Multiply the uncertainty * sensitivity in the final paper, but for the first step we just perturb all terms in the the sensitivity by the same amount. 
- Do nothing for UFlame for now. <br>
- Okafor-sensei has access to the supercomputer for when we run the intensive optimisation scripts <br>
- For the optimisation, we don't want to perturb all the reactions - we only want to perturb a few very specific parameters and reactions. <br>
- For now, we don't know the error ranges - we need to summarise those error ranges in the paper and seperately. 
- We want to optimise for species + LBV simultaneously - need to check optimisation algorithm.

- ICFD submission - aiming for PROCI in December <br>