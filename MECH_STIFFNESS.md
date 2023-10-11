Reaction mechanism stiffness can arise due to several factors. Some common causes of reaction mechanism stiffness include:

1. Fast and Slow Reactions: Reaction mechanisms often consist of both fast and slow reactions. If there is a significant disparity in timescales between these reactions, it can lead to stiffness. The fast reactions can rapidly consume or produce species, causing rapid changes in their concentrations. As a result, the numerical integration of the mechanism becomes challenging.

2. Chemical Timescales: In some cases, chemical reactions can occur on very short timescales, leading to stiffness. For example, radical-radical reactions or reactions involving highly reactive intermediates can have rapid kinetics that require accurate integration methods to capture their behavior.

3. High Activation Energy: Reactions with high activation energy barriers tend to exhibit stiffness. The high energy barrier makes these reactions less likely to occur, but when they do, they can have a significant impact on the overall reaction kinetics. Accurately resolving the behavior of these reactions during integration can be computationally demanding.

4. Coupled Species and Reactions: Stiffness can also arise when reactions involve numerous coupled species. Changes in the concentration of one species can affect multiple reactions, and vice versa. This coupling introduces intricate dependencies within the mechanism, making it more difficult to numerically solve.

5. Numerical Integration Methods: The choice of numerical integration method can influence the stiffness of a reaction mechanism. Explicit integration methods are generally less suitable for stiff systems, as they require small time steps to maintain stability. Implicit integration methods or stiff solvers are often more appropriate for resolving stiff reactions accurately.

6. Inaccurate Rate Constants: If rate constants or reaction rate expressions in the mechanism are inaccurately determined or based on limited experimental data, it can introduce stiffness. Incorrect rate constants can lead to unrealistic timescales for reactions, causing numerical instability.

7. Complexity of the Mechanism: Large and complex reaction mechanisms with numerous species and reactions are more prone to stiffness. The sheer size and complexity of the mechanism can make it challenging to resolve the interdependencies accurately.

It is important to consider these factors when developing or working with reaction mechanisms and to apply appropriate techniques to address stiffness and ensure accurate simulation of combustion processes.

From CnF, 2009 Lu et al: 

The additional obstacle is chemical stiffness induced by highly reactive radicals and fast reversible reactions, which is particularly important when using explicit solvers. In many cases, short species time-scales can be eliminated through the classical quasi-steady-state (QSS) approximation (QSSA) and partial-equilibrium (PE) approximation (PEA)
