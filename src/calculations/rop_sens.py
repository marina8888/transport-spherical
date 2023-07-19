import abc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.settings.filepaths import output_dir

SENSITIVITY_THRESHOLD = 0.1
ROP_THRESHOLD = 0.0000001
PERTURBATION = 1e-2

class BaseFlame(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        """
        Access the same functions relating to ROP and Sensitivity for all flame types by making this an abstract class
        (i.e class cannot exist without its inherited classes being called. The inherited classes are the different flame defintions.)
        """
        pass

    def get_rops(self):
        """
        Plot ROP graphs for each condition and save to csv
        @param self:
        @param species:
        @return:
        """
        try:
            species_ix = self.gas.species_index(self.species)
            x = self.f.grid[:]  # put grids in here [550:750]

            int_rop = []
            net_stoich_coeffs = (self.f.gas.product_stoich_coeffs() - self.f.gas.reactant_stoich_coeffs())

            for r in range(len(self.gas.reaction_equations())):
                ropr = self.f.net_rates_of_progress[r, :]  # put grids in here [r, 550:750]
                rop = net_stoich_coeffs[species_ix, r] * ropr
                int_rop.append(np.trapz(y=rop, x=x))  # use numpy trapezium rule to calculate integral rop values:
            rops_df = pd.DataFrame(index=self.gas.reaction_equations(), columns=["base_case"], data=int_rop)

            self.plot_rop(rops_df)
        except ValueError:
            self.logger.info('Cannot recognise species. Will not plot ROP.')
            pass


    def plot_rop(self, df: pd.DataFrame):
        # sort in order and take main reactions only:
        rop_subset = df[df["base_case"].abs() > ROP_THRESHOLD]
        reactions_above_threshold = (rop_subset.abs().sort_values(by="base_case", ascending=False).index)
        rop_subset.loc[reactions_above_threshold].plot.barh(title=f"Rate of Production for phi = {self.species} at {self.phi}", legend=None)
        plt.rcParams.update({"axes.labelsize": 10})
        plt.gca().invert_yaxis()
        plt.locator_params(axis="x", nbins=6)
        plt.tight_layout()

        plt.savefig(f"{output_dir}/rops/ROP_{type(self).__name__}_{self.species}_{self.blend}_{self.phi}_{self.mech_name}.png")
        plt.show()
        df.to_csv(f"{output_dir}/rops/ROP_{type(self).__name__}_{self.species}_{self.blend}_{self.phi}_{self.mech_name}.csv")


    def get_sens(self):
        """

        @return:
        """
        # take species at outlet or velocity at inlet:
        if self.species == 'lbv':
            print('WARNING: taking lbv for a stagnation flame')
            Su0 = self.f.velocity[0]
        else:
            species_ix = self.gas.species_index(self.species)
            Su0 = self.f.X[species_ix, -1]

        # Create a dataframe to store sensitivity-analysis data:
        sensitivities = pd.DataFrame(index=self.gas.reaction_equations(), columns=["base_case"])

        for m in range(self.gas.n_reactions):
            print(f'reaction {m}')
            self.gas.set_multiplier(1.0)  # reset all multipliers
            self.gas.set_multiplier(1 + PERTURBATION, m)  # perturb reaction m

            # Make sure the grid is not refined, otherwise it won't strictly be a small perturbation analysis
            # Turn auto-mode off since the flame has already been solved
            self.f.solve(loglevel=0, refine_grid=False, auto=False)

            # new values with pertubation:
            if self.species == 'lbv':
                Su = self.f.velocity[0]
            else:
                species_ix = self.gas.species_index(self.species)
                Su = self.f.X[species_ix, -1]

            sensitivities.iloc[m, 0] = (Su - Su0) / (Su0 * PERTURBATION)

        # return mech to normal multipliers:
        self.gas.set_multiplier(1.0)
        self.plot_sens(sensitivities)


    def plot_sens(self, df: pd.DataFrame):
        # sort in order and take main reactions only:
        rop_subset = df[df["base_case"].abs() > SENSITIVITY_THRESHOLD]
        reactions_above_threshold = (rop_subset.abs().sort_values(by="base_case", ascending=False).index)
        rop_subset.loc[reactions_above_threshold].plot.barh(title=f"Sensitivity for {self.species}", legend=None)
        plt.rcParams.update({"axes.labelsize": 10})
        plt.gca().invert_yaxis()
        plt.locator_params(axis="x", nbins=6)
        plt.tight_layout()

        plt.savefig(f"{output_dir}/sens/SENS_{type(self).__name__}_{self.species}_{self.blend}_{self.phi}_{self.mech_name}.png")
        plt.show()
        df.to_csv(f"{output_dir}/sens/SENS_{type(self).__name__}_{self.species}_{self.blend}_{self.phi}_{self.mech_name}.csv")

    def plot_thermo_sens(self):

        # take species at outlet or velocity at inlet:
        if self.species == 'lbv':
            print('WARNING: taking lbv for a stagnation flame')
            Su0 = self.f.velocity[0]
        else:
            species_ix = self.gas.species_index(self.species)
            Su0 = self.f.X[species_ix, -1]

        # Create a dataframe to store sensitivity-analysis data:
        sensitivities = pd.DataFrame(index=self.gas.reaction_equations(), columns=["base_case"])

        for m in range(self.gas.n_reactions):
            self.gas.species()[0].thermo.coeffs = [ 1.00000000e+03, 3.33727920e+00, -4.94024731e-05, 4.99456778e-07,
 -1.79566394e-10, 2.00255376e-14, -9.50158922e+02, -3.20502331e+00, 2.34433112e+00, 7.98052075e-03, -1.94781510e-05, 2.01572094e-08,
 -7.37611761e-12, -9.17935173e+02, 6.83010238e-01]

            # Make sure the grid is not refined, otherwise it won't strictly be a small perturbation analysis
            # Turn auto-mode off since the flame has already been solved
            self.f.solve(loglevel=0, refine_grid=False, auto=False)

            # new values with pertubation:
            if self.species == 'lbv':
                Su = self.f.velocity[0]
            else:
                species_ix = self.gas.species_index(self.species)
                Su = self.f.X[species_ix, -1]

            sensitivities.iloc[m, 0] = (Su - Su0) / (Su0 * PERTURBATION)

        # return mech to normal multipliers:
        self.gas.set_multiplier(1.0)
        self.plot_sens(sensitivities)

        print(self.gas.species()[0].thermo.coeffs)