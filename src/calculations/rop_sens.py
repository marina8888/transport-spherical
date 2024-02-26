import abc
import cantera as ct
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.settings.logger import LogConfig
import src.settings.config_loader as config
logger = LogConfig.configure_logger(__name__)

# code to run sensitivity and rop for a single input species, and all rops for a diagram.

SENSITIVITY_THRESHOLD = config.SENS_PLOT_FILTER
ROP_THRESHOLD = config.ROP_PLOT_FILTER

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
        self.solver_adjoint_init()
        sens_df = self.calculate_solver_adjoint_sens()
        self.plot_sens(sens_df, type_f = 'adjoint_reactions')

    def get_sens_brute_force(self):
        """
        Brute force method to get sensitivities for a specific species + plot
        @return:
        """
        sens_df = self.calculate_brute_force_sens()
        self.plot_sens(sens_df, type_f = 'brute_reactions')

    def get_sens_thermo(self):
        """
        Brute force method to get sensitivities for a specific species + plot
        @return:
        """
        # sensitivity of value of interest to change in entropy and enthalpy:
        # entropy_df, enthalpy_df = self.calculate_thermo()

        # sensitivity of eq coefs to change in entropy and enthalpy:
        # equil_entropy_df, entropy_df = self.calculate_equillibrium_entropy()
        # equil_enthalpy_df, enthalpy_df = self.calculate_equillibrium_enthalpy()
        equil_entropy_df = pd.read_csv("../resources/output/thermo/SENS_EQUIL_ENTHALPY_StagnationFlame_0.02_0.2NH3_1.199407704_okafor-2017.csv")
        equil_enthalpy_df = pd.read_csv("../resources/output/thermo/SENS_EQUIL_ENTHALPY_StagnationFlame_0.02_0.2NH3_1.199407704_okafor-2017.csv")
        enthalpy_df = pd.read_csv("../resources/output/thermo/SENS_NO_SP_ENTHALPY_StagnationFlame_0.02_0.2NH3_1.199407704_okafor-2017.csv")
        entropy_df = pd.read_csv("../resources/output/thermo/SENS_NO_SP_ENTROPY_StagnationFlame_0.02_0.2NH3_1.199407704_okafor-2017.csv")

        # solve:
        sens_df = self.solve_matrix(equil_entropy_df, equil_enthalpy_df, entropy_df, enthalpy_df, )
        self.plot_sens(sens_df, type_f='thermo_sol')

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
        all_rops_df.to_csv(f"{config.OUTPUT_DIR_ROP}/ROP_{type(self).__name__}_ALLSPECIES_{self.blend}_{self.phi}_{self.mech_name}.csv")

    def get_rop_distance(self):
        """
        Get the ROP data for all species in order to plot a ROP diagram
        @return:
        """
        print(f'doing rop distance for {self.species}')
        rops_df = self.calculate_rops_distance()
        print(rops_df)
        rops_df.to_csv(
            f"{config.OUTPUT_DIR_ROP}/rops/ROP_{type(self).__name__}_DIST_{self.blend}_{self.species}_{self.phi}_{self.mech_name}.csv")

    # end of get functions.
    # -----------------------------------------------------------------------------------------------------------------#
    # functions defintions below:

    def solver_adjoint_init(self):
        """
        Setup for getting solver adjoint values
        @return:
        """
        self.grid_point = min(range(len(self.f.grid)), key=lambda i: abs(self.f.grid[i] - config.SENS_SPECIES_LOC))
        logger.info(f"using an adjoint method on grid point {self.grid_point}")

        # Index of self.species in the global solution vector
        if self.species == 'lbv':
            # auto target grid point 0 in the solution vector for lbv:
            i_spec = self.f.inlet.n_components + self.f.flame.component_index('velocity')
            self.spec_0 = self.f.velocity[0]
            logger.info(f"running sensitivity analysis for lbv")

        else:
            # specify species and which grid point to target in solution vector:
            i_spec = self.f.inlet.n_components + self.f.flame.component_index(self.species) + self.f.domains[1].n_components*self.grid_point
            self.spec_0 = self.f.X[self.gas.species_index(self.species), self.grid_point]
            logger.info(f"running sensitivity analysis for species {self.species}")

        # ImpingingJet flame has three domains (inlet, flame, surface), but only the flame domain stores information:
        Nvars = sum(D.n_components * D.n_points for D in self.f.domains)

        self.dgdx = np.zeros(Nvars)
        self.dgdx[i_spec] = config.SENS_PERTURBATION


    def calculate_rops(self, species = None):
        """
        Calculate ROP values for species selected in the flame call.
        @param self:
        @param species:
        @return: None
        """
        self.from_grid_point = min(range(len(self.f.grid)), key=lambda i: abs(self.f.grid[i] - config.ROP_LOC_FROM))
        self.to_grid_point = min(range(len(self.f.grid)), key=lambda i: abs(self.f.grid[i] - config.ROP_LOC_TO))

        if species is None:
            species_ix = self.gas.species_index(self.species) # use the default species passed in
            columns = 'base_case'

        else:
            species_ix = self.gas.species_index(species) # use the species passed in as a loop for all ROPs
            columns = species

        try:
            x = self.f.grid[self.from_grid_point:self.to_grid_point]  # put grids in here [550:750] to select specific flame zone
            int_rop = []
            net_stoich_coeffs = self.f.gas.product_stoich_coeffs - self.f.gas.reactant_stoich_coeffs

            for r in range(len(self.gas.reaction_equations())):
                ropr = self.f.net_rates_of_progress[r, self.from_grid_point:self.to_grid_point]  # put grids in here [r, 550:750]
                rop = net_stoich_coeffs[species_ix, r] * ropr
                int_rop.append(np.trapz(y=rop, x=x))  # use numpy trapezium rule to calculate integral rop values:
            rops_df = pd.DataFrame(index=self.gas.reaction_equations(), columns=[columns], data=int_rop)
            return rops_df

        except ValueError:
            self.logger.info('Cannot recognise species. Cannot calculate ROP')
            pass

    def calculate_rops_distance(self, species = None):
        """
        Calculate ROP values for species selected in the flame call.
        @param self:
        @param species:
        @return: None
        """
        self.from_grid_point = min(range(len(self.f.grid)), key=lambda i: abs(self.f.grid[i] - config.ROP_LOC_FROM))
        self.to_grid_point = min(range(len(self.f.grid)), key=lambda i: abs(self.f.grid[i] - config.ROP_LOC_TO))

        if species is None:
            species_ix = self.gas.species_index(self.species) # use the default species passed in

        else:
            species_ix = self.gas.species_index(species) # use the species passed in as a loop for all ROPs
            columns = species

        try:
            x = self.f.grid[self.from_grid_point:self.from_grid_point]  # put grids in here [550:750] to select specific flame zone
            all_rop = []
            net_stoich_coeffs = self.f.gas.product_stoich_coeffs - self.f.gas.reactant_stoich_coeffs

            for r in range(len(self.gas.reaction_equations())):
                ropr = self.f.net_rates_of_progress[r, self.from_grid_point:self.from_grid_point]  # put grids in here [r, 550:750]
                rop = net_stoich_coeffs[species_ix, r] * ropr
                all_rop.append(rop)
            rops_df = pd.DataFrame(index=self.gas.reaction_equations(), columns=x, data=all_rop)
            return rops_df

        except ValueError:
            self.logger.info('Cannot recognise species. Cannot calculate ROP')
            pass


    def calculate_brute_force_sens(self):
        """
        Calculate sensitivities for a species wrt thermo using the brute force method
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
            self.gas.set_multiplier(1 + config.SENS_PERTURBATION, m)  # perturb reaction m

            # Make sure the grid is not refined, otherwise it won't strictly be a small perturbation analysis
            # Turn auto-mode off since the flame has already been solved
            self.f.solve(loglevel=0, refine_grid=False, auto=False)

            # new values with pertubation:
            if self.species == 'lbv':
                Su = self.f.velocity[0]
            else:
                species_ix = self.gas.species_index(self.species)
                Su = self.f.X[species_ix, -1]

            sens_df.iloc[m, 0] = (Su - Su0) / (Su0 * config.SENS_PERTURBATION)
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
        def perturb(sim, i, dp):
            sim.gas.set_multiplier(1 + dp, i)

        sens_vals = self.f.solve_adjoint(perturb, self.gas.n_reactions, self.dgdx) / self.spec_0
        return pd.DataFrame(index=self.gas.reaction_equations(), columns=['base_case'], data = sens_vals)

    def calculate_thermo(self):
        self.solver_adjoint_init()

        def perturb_enthalpy(sim, i, dp):
            S = sim.gas.species(i)
            print(f"running sensitivity wrt to entropy of: {S}")
            st = S.thermo
            coeffs = st.coeffs
            coeffs[[6, 13]] += 601.39
            snew = ct.NasaPoly2(st.min_temp, st.max_temp, st.reference_pressure, coeffs)
            S.thermo = snew
            sim.gas.modify_species(sim.gas.species_index(i), S)

        def perturb_entropy(sim, i, dp):
            S = sim.gas.species(i)
            print(f"running sensitivity wrt to entropy of: {S}")
            st = S.thermo
            coeffs = st.coeffs
            coeffs[[7, 14]] += 1.2027
            snew = ct.NasaPoly2(st.min_temp, st.max_temp, st.reference_pressure, coeffs)
            S.thermo = snew
            sim.gas.modify_species(sim.gas.species_index(i), S)

        entropy_df = pd.DataFrame(index = self.gas.species_names)
        enthalpy_df = pd.DataFrame(index = self.gas.species_names)

        entropy_df[self.species] = self.f.solve_adjoint(perturb_entropy, len(self.gas.species()), self.dgdx) / self.spec_0
        enthalpy_df[self.species] = self.f.solve_adjoint(perturb_enthalpy, len(self.gas.species()), self.dgdx) / self.spec_0
        enthalpy_df.to_csv(f"{config.OUTPUT_DIR_THERMO}/SENS_{self.species}_ENTHALPY_{type(self).__name__}_{config.SENS_SPECIES_LOC}_{self.blend}_{self.phi}_{self.mech_name}.csv")
        entropy_df.to_csv(f"{config.OUTPUT_DIR_THERMO}/SENS_{self.species}_ENTROPY_{type(self).__name__}_{config.SENS_SPECIES_LOC}_{self.blend}_{self.phi}_{self.mech_name}.csv")
        return entropy_df, enthalpy_df



    def calculate_solver_adjoint_sens_thermo(self):
        """
        Compute the normalized sensitivities of the species production, taken at the final grid point
        :math:`s_{i, spec}` with respect to the reaction rate constants :math:`k_i`:
        .. math::
            s_{i, spec} = \frac{k_i}{[X]} \frac{d[X]}{dk_i}
        @return:
        """
        def perturb(sim, i, dp):
            sim.gas.set_multiplier(1 + dp, i)

        sens_vals = self.f.solve_adjoint(perturb, self.gas.n_reactions, self.dgdx) / self.spec_0
        return pd.DataFrame(index=self.gas.reaction_equations(), columns=['base_case'], data = sens_vals)

    def calculate_thermo(self):
        self.solver_adjoint_init()

        def perturb_enthalpy(sim, i, dp):
            S = sim.gas.species(i)
            print(f"running sensitivity wrt to entropy of: {S}")
            st = S.thermo
            coeffs = st.coeffs
            coeffs[[6, 13]] += 601.39
            snew = ct.NasaPoly2(st.min_temp, st.max_temp, st.reference_pressure, coeffs)
            S.thermo = snew
            sim.gas.modify_species(sim.gas.species_index(i), S)

        def perturb_entropy(sim, i, dp):
            S = sim.gas.species(i)
            print(f"running sensitivity wrt to entropy of: {S}")
            st = S.thermo
            coeffs = st.coeffs
            coeffs[[7, 14]] += 1.2027
            snew = ct.NasaPoly2(st.min_temp, st.max_temp, st.reference_pressure, coeffs)
            S.thermo = snew
            sim.gas.modify_species(sim.gas.species_index(i), S)

        entropy_df = pd.DataFrame(index = self.gas.species_names)
        enthalpy_df = pd.DataFrame(index = self.gas.species_names)

        entropy_df[self.species] = self.f.solve_adjoint(perturb_entropy, len(self.gas.species()), self.dgdx) / self.spec_0
        enthalpy_df[self.species] = self.f.solve_adjoint(perturb_enthalpy, len(self.gas.species()), self.dgdx) / self.spec_0
        enthalpy_df.to_csv(f"{config.OUTPUT_DIR_THERMO}/SENS_{self.species}_ENTHALPY_{type(self).__name__}_{config.SENS_SPECIES_LOC}_{self.blend}_{self.phi}_{self.mech_name}.csv")
        entropy_df.to_csv(f"{config.OUTPUT_DIR_THERMO}/SENS_{self.species}_ENTROPY_{type(self).__name__}_{config.SENS_SPECIES_LOC}_{self.blend}_{self.phi}_{self.mech_name}.csv")
        return entropy_df, enthalpy_df

    def calculate_equillibrium_enthalpy(self):
        """
        Sensitivity of reaction equilirbium coefficients to change in entropy and enthalpy using brute force
        @return: sensitivity dataframe
        """
        equil_sens_df = pd.DataFrame(data = self.gas.equilibrium_constants, columns=["base_case"], index=self.gas.reaction_equations())
        sens_sp_df = pd.DataFrame(columns=[self.species], index=self.gas.species_names)

        # take species at outlet or velocity at inlet:
        if self.species == 'lbv':
            print('WARNING: taking lbv for a stagnation flame')
            Su0 = self.f.velocity[0]
        else:
            species_ix = self.gas.species_index(self.species)
            Su0 = self.f.X[species_ix, -1]


        for s in range(len(self.gas.species())):
            species = self.gas.species(s)
            species_thermo_original = species.thermo
            species_thermo = species.thermo
            coeffs = species_thermo.coeffs
            coeffs[[6, 13]] += 601.39
            nasa_new = ct.NasaPoly2(species_thermo.min_temp, species_thermo.max_temp, species_thermo.reference_pressure, coeffs)
            species.thermo = nasa_new
            self.gas.modify_species(self.gas.species_index(s), species)

            # Make sure the grid is not refined, and auto off also, otherwise it won't strictly be a small perturbation analysis
            self.f.solve(loglevel=0, refine_grid=False, auto=False)

            # new values with pertubation:
            if self.species == 'lbv':
                Su = self.f.velocity[0]
            else:
                species_ix = self.gas.species_index(self.species)
                Su = self.f.X[species_ix, -1]

            sens_sp_df.iloc[s, 0] = (Su - Su0) / (Su0 * 5)
            equil_sens_df[species] = self.gas.equilibrium_constants

            print(sens_sp_df)
            print(equil_sens_df)
            # return coefficients to their original values:
            species.thermo = species_thermo_original
            self.gas.modify_species(self.gas.species_index(s), species)

        sens_sp_df.to_csv(
            f"{config.OUTPUT_DIR_THERMO}/SENS_{self.species}_SP_ENTROPY_{type(self).__name__}_{config.SENS_SPECIES_LOC}_{self.blend}_{self.phi}_{self.mech_name}.csv")
        equil_sens_df.to_csv(f"{config.OUTPUT_DIR_THERMO}/SENS_EQUIL_ENTROPY_{type(self).__name__}_{config.SENS_SPECIES_LOC}_{self.blend}_{self.phi}_{self.mech_name}.csv")
        return equil_sens_df, sens_sp_df

    def calculate_equillibrium_entropy(self):
        """
        Sensitivity of reaction equilirbium coefficients to change in entropy and enthalpy using brute force
        @return: sensitivity dataframe
        """
        equil_sens_df = pd.DataFrame(data = self.gas.equilibrium_constants, columns=["base_case"], index=self.gas.reaction_equations())
        sens_sp_df = pd.DataFrame(columns=[self.species], index=self.gas.species_names)

        print(sens_sp_df)
        print(equil_sens_df)

        # take species at outlet or velocity at inlet:
        if self.species == 'lbv':
            print('WARNING: taking lbv for a stagnation flame')
            Su0 = self.f.velocity[0]
        else:
            species_ix = self.gas.species_index(self.species)
            Su0 = self.f.X[species_ix, -1]

        for s in range(len(self.gas.species())):
            species = self.gas.species(s)
            species_thermo_original = species.thermo
            species_thermo = species.thermo
            coeffs = species_thermo.coeffs
            coeffs[[7, 14]] += 1.2027
            nasa_new = ct.NasaPoly2(species_thermo.min_temp, species_thermo.max_temp, species_thermo.reference_pressure,
                                    coeffs)
            species.thermo = nasa_new
            self.gas.modify_species(self.gas.species_index(s), species)

            # Make sure the grid is not refined, and auto off also, otherwise it won't strictly be a small perturbation analysis
            self.f.solve(loglevel=0, refine_grid=False, auto=False)

            # new values with pertubation:
            if self.species == 'lbv':
                Su = self.f.velocity[0]
            else:
                species_ix = self.gas.species_index(self.species)
                Su = self.f.X[species_ix, -1]
            sens_sp_df.iloc[s, 0] = (Su - Su0) / (Su0 * 5)
            equil_sens_df[species] = self.gas.equilibrium_constants

            print(sens_sp_df)
            print(equil_sens_df)
            # return coefficients to their original values:
            species.thermo = species_thermo_original
            self.gas.modify_species(self.gas.species_index(s), species)

        sens_sp_df.to_csv(
            f"{config.OUTPUT_DIR_THERMO}/SENS_{self.species}_SP_ENTHALPY_{type(self).__name__}_{config.SENS_SPECIES_LOC}_{self.blend}_{self.phi}_{self.mech_name}.csv")
        equil_sens_df.to_csv(
            f"{config.OUTPUT_DIR_THERMO}/SENS_EQUIL_ENTHALPY_{type(self).__name__}_{config.SENS_SPECIES_LOC}_{self.blend}_{self.phi}_{self.mech_name}.csv")
        return equil_sens_df, sens_sp_df

    def solve_matrix(self, df_eq_constants, df_eq_constants2, df_species, df_species2):
        """

        @param df_eq_constants: sensitivity of equilibruim constants of each reaction
        @param df_species: sensitivity of value of interest
        @return:
        """
        print(df_eq_constants.columns.values, df_eq_constants2.columns.values, df_species.columns.values, df_species2.columns.values)
        print(df_eq_constants)
        # normalise the data of reaction constants:
        columns_to_divide = [col for col in df_eq_constants.columns.values if col not in ['Unnamed: 0', 'base_case']]
        df_eq_constants[columns_to_divide] = df_eq_constants[columns_to_divide].sub(df_eq_constants['base_case'], axis=0)
        df_eq_constants[columns_to_divide] = df_eq_constants[columns_to_divide].div(df_eq_constants['base_case'], axis=0)
        # Define the coefficient matrix A and the constants vector b
        A_array_list = [column_data.values for column_name, column_data in df_eq_constants[columns_to_divide].items()]
        print(A_array_list)
        # normalise the data of reaction constants:
        columns_to_divide = [col for col in df_eq_constants2.columns.values if col not in ['Unnamed: 0', 'base_case']]
        df_eq_constants2[columns_to_divide] = df_eq_constants2[columns_to_divide].sub(df_eq_constants2['base_case'], axis=0)
        df_eq_constants2[columns_to_divide] = df_eq_constants2[columns_to_divide].div(df_eq_constants2['base_case'], axis=0)

        # Define the coefficient matrix A and the constants vector b
        A_array_list2 = [column_data.values for column_name, column_data in df_eq_constants2[columns_to_divide].items()]


        # Convert the list of arrays to a NumPy array
        A = np.vstack((np.array(A_array_list), np.array(A_array_list2)))
        b = np.hstack((df_species[self.species].to_numpy(), df_species2[self.species].to_numpy()))
        print(A, b)

        alpha = 0.00000000001  # regularization parameter
        # x = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ b)
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=alpha)
        # x = np.linalg.solve(A, b)

        print(f"residuals are {residuals}")
        x = pd.DataFrame(index = self.gas.reaction_equations(), data = x, columns = ['base_case'])
        x = x[~x.index.duplicated(keep='first')].copy()
        logger.info(f"A is {A}\n b is {b}\n solution is {x}\n")

        x.to_csv(f"{config.OUTPUT_DIR_THERMO}/SENS_SOLUTION_{type(self).__name__}_{config.SENS_SPECIES_LOC}_{self.species}_{self.blend}_{self.phi}_{self.mech_name}.csv")
        return x

    def plot_sens(self, df: pd.DataFrame, type_f = 'reactions'):
        """
        Plot sensitivity for a specific species
        @param df:
        @return:
        """
        # try:
        #     # sort in order and take main reactions only:
        #     rop_subset = df[df["base_case"].abs() > config.SENS_PLOT_FILTER]
        #     reactions_above_threshold = (rop_subset.abs().sort_values(by="base_case", ascending=False).index)
        #     rop_subset.loc[reactions_above_threshold].plot.barh(title=f"Sensitivity for {self.species} at phi = {self.phi}", legend=None)
        #     plt.rcParams.update({"axes.labelsize": 12})
        #     plt.gca().invert_yaxis()
        #     plt.locator_params(axis="x", nbins=6)
        #     plt.tight_layout()
        #     plt.show()
        #     plt.savefig(f"{config.GRAPHS_DIR_SENS}/SENS_{type_f}_{type(self).__name__}_{config.SENS_SPECIES_LOC}_{self.species}_{self.blend}_{self.phi}_{self.mech_name}.png")
        #
        # except IndexError:
        #     logger.info("Please adjust threshold or perturbation values")

        df.to_csv(f"{config.OUTPUT_DIR_SENS}/SENS_{type_f}_{type(self).__name__}_{config.SENS_SPECIES_LOC}_{self.species}_{self.blend}_{self.phi}_{self.mech_name}.csv")


    def plot_rop(self, df: pd.DataFrame):
        """
        Plot ROP for a specific species.
        @param df:
        @return:
        """
        try:
            # sort in order and take main reactions only:
            rop_subset = df[df["base_case"].abs() > config.ROP_PLOT_FILTER]
            reactions_above_threshold = (rop_subset.abs().sort_values(by="base_case", ascending=False).index)
            rop_subset.loc[reactions_above_threshold].plot.barh(title=f"Rate of production for {self.species} at phi = {self.phi}", legend=None)
            plt.rcParams.update({"axes.labelsize": 10})
            plt.gca().invert_yaxis()
            plt.locator_params(axis="x", nbins=6)
            plt.tight_layout()
            plt.show()
            plt.savefig(f"{config.GRAPHS_DIR_ROP}/ROP_{type(self).__name__}_{config.ROP_LOC_FROM}-{config.ROP_LOC_TO}_{self.species}_{self.blend}_{self.phi}_{self.mech_name}.png")

        except IndexError:
            logger.info("Please adjust threshold or perturbation values")

        df.to_csv(f"{config.OUTPUT_DIR_ROP}/ROP_{type(self).__name__}_{config.ROP_LOC_FROM}-{config.ROP_LOC_TO}_{self.species}_{self.blend}_{self.phi}_{self.mech_name}.csv")
