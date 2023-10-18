import abc
import cantera as ct
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.settings.logger import LogConfig
from src.settings.filepaths import output_dir
logger = LogConfig.configure_logger(__name__)

# code to run sensitivity and rop for a single input species, and all rops for a diagram.

SENSITIVITY_THRESHOLD = 0.2
SENSITIVITY_THRESHOLD_ADJOINT = 0.00001
ROP_THRESHOLD = 0.0001
BRUTE_FORCE_PERTURBATION = 1e-2
ADJOINT_PERTURBATION = 0.05
SENSITIVITY_POSITION = -1

class BaseFlame(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        """
        Access the same functions relating to ROP and Sensitivity for all flame types by making this an abstract class
        (i.e class cannot exist without its inherited classes being called. The inherited classes are the different flame defintions.)
        """
        pass

    def get_rop(self):
        """
        General function to get and plot ROPs
        @return: None
        """
        rops_df = self.calculate_rops()
        self.plot_rop(rops_df)

    def get_sens_adjoint(self):
        """
        Adjoint method to get sensitivities for a specific species + plot
        @return: None
        """
        sens_df = self.calculate_solver_adjoint_sens()
        print("plotting adjoint")
        self.plot_sens(sens_df)

    def get_sens_brute_force(self):
        """
        Brute force method to get sensitivities for a specific species + plot
        @return:
        """
        sens_df = self.calculate_brute_force_sens()
        print("plotting brute force")
        self.plot_sens(sens_df)

    def get_sens_thermo(self):
        """
        Brute force method to get sensitivities for a specific species + plot
        @return:
        """
        sens_df = self.calculate_sens_thermo()
        print("plotting adjoint thermo")
        self.plot_sens(sens_df)
    def get_rop_all(self):
        """
        Get the ROP data for all species in order to plot a ROP diagram
        @return:
        """
        all_rops_df = pd.DataFrame(index = self.gas.reaction_equations())
        for s in self.gas.species_names:
            logger.info(f"running all ROPs for species: {s}")
            temp_df = self.calculate_rops(s)
            all_rops_df = pd.concat([all_rops_df, temp_df], axis = 1)
        all_rops_df.to_csv(f"{output_dir}/rops/ROP_{type(self).__name__}_ALLSPECIES_{self.blend}_{self.phi}_{self.mech_name}.csv")

    def calculate_rops(self, species = None):
        """
        Calculate ROP values for species selected in the flame call.
        @param self:
        @param species:
        @return: None
        """
        if species is None:
            species_ix = self.gas.species_index(self.species) # use the default species passed in
            columns = 'base_case'

        else:
            species_ix = self.gas.species_index(species) # use the species passed in as a loop for all ROPs
            columns = species

        try:
            x = self.f.grid[:]  # put grids in here [550:750] to select specific flame zone
            int_rop = []
            net_stoich_coeffs = (self.f.gas.product_stoich_coeffs() - self.f.gas.reactant_stoich_coeffs())

            for r in range(len(self.gas.reaction_equations())):
                ropr = self.f.net_rates_of_progress[r, :]  # put grids in here [r, 550:750]
                rop = net_stoich_coeffs[species_ix, r] * ropr
                int_rop.append(np.trapz(y=rop, x=x))  # use numpy trapezium rule to calculate integral rop values:
            rops_df = pd.DataFrame(index=self.gas.reaction_equations(), columns=[columns], data=int_rop)
            return rops_df

        except ValueError:
            self.logger.info('Cannot recognise species. Cannot calculate ROP')
            pass


    def calculate_brute_force_sens(self):
        """
        Calculate sensitivities for a species using the brute force method
        @return: sensitivity dataframe
        """
        # Create a dataframe to store sensitivity-analysis data:
        sens_df = pd.DataFrame(index=self.gas.reaction_equations(), columns=["base_case"])

        # take species at outlet or velocity at inlet:
        if self.species == 'lbv':
            print('WARNING: taking lbv for a stagnation flame')
            Su0 = self.f.velocity[0]
        else:
            species_ix = self.gas.species_index(self.species)
            Su0 = self.f.X[species_ix, -1]

        for m in range(self.gas.n_reactions):
            print(f'reaction {m}')
            self.gas.set_multiplier(1.0)  # reset all multipliers
            self.gas.set_multiplier(1 + BRUTE_FORCE_PERTURBATION, m)  # perturb reaction m

            # Make sure the grid is not refined, otherwise it won't strictly be a small perturbation analysis
            # Turn auto-mode off since the flame has already been solved
            self.f.solve(loglevel=0, refine_grid=False, auto=False)

            # new values with pertubation:
            if self.species == 'lbv':
                Su = self.f.velocity[0]
            else:
                species_ix = self.gas.species_index(self.species)
                Su = self.f.X[species_ix, -1]

            sens_df.iloc[m, 0] = (Su - Su0) / (Su0 * BRUTE_FORCE_PERTURBATION)
        # return mech to normal multipliers:
        self.gas.set_multiplier(1.0)
        return sens_df

    def calculate_solver_adjoint_sens(self):
        """
        Compute the normalized sensitivities of the species production, taken at the final grid point
        :math:`s_{i, spec}` with respect to the reaction rate constants :math:`k_i`:
        .. math::
            s_{i, spec} = \frac{k_i}{[X]} \frac{d[X]}{dk_i}
        @return:
        """
        # components are accessible values of interest, and n_points is number of grid points.
        # ImpingingJet flame has three domains (inlet, flame, surface), but only the flame domain stores information
        Nvars = sum(D.n_components * D.n_points for D in self.f.domains)
        if SENSITIVITY_POSITION == -1:
            grid_point = len(self.f.grid) - 1 # gets the final grid point
        else:
            grid_point = SENSITIVITY_POSITION  # gets the final grid point

        # Index of self.species in the global solution vector
        # i_spec = self.f.inlet.n_components + self.f.flame.component_index(self.species)
        i_spec = self.f.inlet.n_components + self.f.flame.component_index(self.species) + self.f.domains[1].n_components*grid_point

        dgdx = np.zeros(Nvars)
        dgdx[i_spec] = ADJOINT_PERTURBATION
        spec_0 = self.f.X[self.gas.species_index(self.species), SENSITIVITY_POSITION]

        def perturb(sim, i, dp):
            print(i)
            sim.gas.set_multiplier(1 + dp, i)

        sens_vals = self.f.solve_adjoint(perturb, self.gas.n_reactions, dgdx) / spec_0
        return pd.DataFrame(index=self.gas.reaction_equations(), columns=['base_case'], data = sens_vals)

    def calculate_sens_thermo(self):
        """
        Modification on the solver adjoint function to perturb thermo parameters.
        @return:
        """
        # components are accessible values of interest, and n_points is number of grid points.
        # ImpingingJet flame has three domains (inlet, flame, surface), but only the flame domain stores information
        Nvars = sum(D.n_components * D.n_points for D in self.f.domains)

        if SENSITIVITY_POSITION == -1:
            grid_point = len(self.f.grid) - 1 # gets the final grid point
        else:
            grid_point = SENSITIVITY_POSITION  # gets the final grid point

        # Index of self.species in the global solution vector
        # i_spec = self.f.inlet.n_components + self.f.flame.component_index(self.species)
        i_spec = self.f.inlet.n_components + self.f.flame.component_index(self.species) + self.f.domains[1].n_components*grid_point

        dgdx = np.zeros(Nvars)
        dgdx[i_spec] = ADJOINT_PERTURBATION
        spec_0 = self.f.X[self.gas.species_index(self.species), SENSITIVITY_POSITION]

        def perturb(sim, i, dp):
            S = sim.gas.species(i)
            st = S.thermo
            coeffs = st.coeffs
            coeffs[[6, 13]] += 5 / ct.gas_constant
            snew = ct.NasaPoly2(st.min_temp, st.max_temp, st.reference_pressure, coeffs)
            S.thermo = snew
            sim.gas.modify_species(sim.gas.species_index(i), S)
        print(self.gas.species())
        print(len(self.gas.species()))
        sens_vals = self.f.solve_adjoint(perturb, len(self.gas.species()), dgdx) / spec_0
        return pd.DataFrame(index=self.gas.species(), columns=['base_case'], data = sens_vals)


        # Create a dataframe to store sensitivity-analysis data:



    def calculate_sens_trans(self):
        pass

    def plot_sens(self, df: pd.DataFrame):
        """
        Plot sensitivity for a specific species
        @param df:
        @return:
        """
        try:
            # sort in order and take main reactions only:
            rop_subset = df[df["base_case"].abs() > SENSITIVITY_THRESHOLD_ADJOINT]
            reactions_above_threshold = (rop_subset.abs().sort_values(by="base_case", ascending=False).index)
            rop_subset.loc[reactions_above_threshold].plot.barh(title=f"Sensitivity for {self.species} at phi = {self.phi}", legend=None)
            plt.rcParams.update({"axes.labelsize": 12})
            plt.gca().invert_yaxis()
            plt.locator_params(axis="x", nbins=6)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/sens/SENS_{type(self).__name__}_{self.species}_{self.blend}_{self.phi}_{self.mech_name}.png")
            plt.show()

        except IndexError:
            logger.info("Please adjust threshold or perturbation values")

        df.to_csv(f"{output_dir}/sens/SENS_{type(self).__name__}_{self.species}_{self.blend}_{self.phi}_{self.mech_name}.csv")


    def plot_rop(self, df: pd.DataFrame):
        """
        Plot ROP for a specific species.
        @param df:
        @return:
        """
        try:
            # sort in order and take main reactions only:
            rop_subset = df[df["base_case"].abs() > ROP_THRESHOLD]
            reactions_above_threshold = (rop_subset.abs().sort_values(by="base_case", ascending=False).index)
            rop_subset.loc[reactions_above_threshold].plot.barh(title=f"Rate of production for {self.species} at phi = {self.phi}", legend=None)
            plt.rcParams.update({"axes.labelsize": 10})
            plt.gca().invert_yaxis()
            plt.locator_params(axis="x", nbins=6)
            plt.tight_layout()

            plt.savefig(f"{output_dir}/rops/ROP_{type(self).__name__}_{self.species}_{self.blend}_{self.phi}_{self.mech_name}.png")
            plt.show()

        except IndexError:
            logger.info("Please adjust threshold or perturbation values")

        df.to_csv(f"{output_dir}/rops/ROP_{type(self).__name__}_{self.species}_{self.blend}_{self.phi}_{self.mech_name}.csv")
