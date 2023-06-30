import cantera as ct
from src.settings.filepaths import mech_dir, output_dir, output_dir_numerical
from src.plotter.rop_sens import plot_rop
import pandas as pd
from src.settings.logger import LogConfig
import os
import numpy as np
import matplotlib.pyplot  as plt

# this file runs a stagnation stabilised flame in Cantera
SENSITIVITY_THRESHOLD = 0.01
ROP_THRESHHOLD = 0.01

class StagnationFlame:
    def __init__(self, oxidizer, blend, fuel, phi, T_in, P, T, vel, flash_point, mech_name, species = None):
        self.oxidizer = oxidizer
        self.blend = blend
        self.fuel = fuel
        self.phi = phi
        self.T_in = T_in
        self.P = P
        self.TP = (T_in, P)
        self.T = T
        self.vel = vel
        self.species = species
        self.flash_point = flash_point
        self.mech_name = mech_name
        self.logger = LogConfig.configure_logger(__name__)

    def configure_gas(self):
        # gas is a solution class from the Cantera library
        self.gas = ct.Solution(f"{mech_dir}/{self.mech_name}.cti")
        self.gas.TP = self.TP
        self.gas.set_equivalence_ratio(self.phi, fuel=self.fuel, oxidizer=self.oxidizer)

    def configure_flame(self):
        # we are using an ImpingingJet class but there are others that might be suitable for other experiments
        self.f = ct.ImpingingJet(gas=self.gas, width=0.02)
        self.f.set_max_grid_points(domain=1, npmax=1200)
        self.f.inlet.mdot = self.vel * self.gas.density
        self.f.surface.T = self.T
        self.f.transport_model = "Multi"
        self.f.soret_enabled = True
        self.f.radiation_enabled = False
        self.f.set_initial_guess("equil")  # assume adiabatic equilibrium products
        self.f.set_refine_criteria(ratio=3, slope=0.012, curve=0.028, prune=0.0001)
        # self.f.set_refine_criteria(ratio=3, slope=0.05, curve=0.1, prune=0)

    def check_solution_file_exists(self, filename, columns):
        if not (os.path.exists(filename)):
            pd.DataFrame(columns=columns).to_csv(f"{filename}")


    def solve(self):

        try:
            self.f.solve(loglevel=1, auto=True)
            if max(self.f.T) < self.flash_point:
                return 0
            else:
                data_x = {
                    "phi": self.phi,
                    "grid": len(self.f.grid),
                    "T_in": self.T_in,
                    "T": self.T,
                    "P": self.P,
                    "vel": self.vel,
                    "blend": self.blend,
                    "fuel": self.fuel,
                    "oxidizer": self.oxidizer,
                }
                data_y = dict(zip(self.gas.species_names, self.f.X[:, -1]))
                data = {**data_x, **data_y}
                df = pd.json_normalize(data)
                filename = f"{self.blend}_{self.mech_name}.csv"

                self.check_solution_file_exists(filename, df.columns)
                df.to_csv(f"{filename}", mode="a", header=False)

        except ct.CanteraError:
            pass

    def get_rops(self, species:str):
        """
        Plot ROP graphs for each condition and save to csv
        @param self:
        @param species:
        @return:
        """
        species_ix = self.gas.species_index(species)
        x = self.f.grid[:]  # put grids in here [550:750]

        int_rop = []
        net_stoich_coeffs = (self.f.gas.product_stoich_coeffs() - self.f.gas.reactant_stoich_coeffs())

        for r in range(len(self.gas.reaction_equations())):
            ropr = self.f.net_rates_of_progress[r, :]  # put grids in here [r, 550:750]
            rop = net_stoich_coeffs[species_ix, r] * ropr
            int_rop.append(np.trapz(y=rop, x=x))  # use numpy trapezium rule to calculate integral rop values:
        rops_df = pd.DataFrame(index=self.gas.reaction_equations(), columns=["base_case"], data=int_rop)

        self.plot_rop(rops_df)

    def plot_rop(self, df: pd.DataFrame):
        # sort in order and take main reactions only:
        rop_subset = df[df["base_case"].abs() > ROP_THRESHHOLD]
        reactions_above_threshold = (
            rop_subset.abs().sort_values(by="base_case", ascending=False).index
        )

        rop_subset.loc[reactions_above_threshold].plot.barh(
            title=f"Rate of Production for {self.species}", legend=None
        )
        # set ticks on the x axis to stop overlap:

        plt.rcParams.update({"axes.labelsize": 10})
        plt.gca().invert_yaxis()
        plt.locator_params(axis="x", nbins=6)
        plt.tight_layout()
        plt.savefig()
        plt.show()
        df.to_csv(f"{output_dir}/rops/ROP_{self.species}_{self.blend}_{self.phi}_{self.mech_name}.csv")


    # def get_sensitivities(self, species=None):
    #     """
    #
    #     @param self:
    #     @param species:
    #     @return:
    #     """
    #
    #     Su = None
    #     Su0 = None
    #     if species == None:
    #         # take lbv sense by default:
    #         Su0 = self.f.velocity[0]
    #     else:
    #         print(f"doing sensitivity analysis for {species}")
    #         # take species at outlet:
    #         species_ix = gas.species_index(species)
    #         Su0 = f.X[species_ix, -1]
    #
    #     # Create a dataframe to store sensitivity-analysis data
    #     sensitivities = pd.DataFrame(
    #         index=gas.reaction_equations(), columns=["base_case"]
    #     )
    #
    #     for m in range(gas.n_reactions):
    #         gas.set_multiplier(1.0)  # reset all multipliers
    #         gas.set_multiplier(1 + self.pertubation, m)  # perturb reaction m
    #
    #         # Always force loglevel=0 for this
    #         # Make sure the grid is not refined, otherwise it won't strictly
    #         # be a small perturbation analysis
    #         # Turn auto-mode off since the flame has already been solved
    #         f.solve(loglevel=0, refine_grid=False, auto=False)
    #
    #         # new values with pertubation:
    #         if species == None:
    #             # take lbv sense by default:
    #             Su = f.velocity[0]
    #         else:
    #             print(f"doing sensitivity analysis for {species}")
    #             # take species at outlet:
    #             species_ix = gas.species_index(species)
    #             Su = f.X[species_ix, -1]
    #
    #         sensitivities.iloc[m, 0] = (Su - Su0) / (Su0 * self.pertubation)
    #     # return mech to normal:
    #     gas.set_multiplier(1.0)
    #
    #     return sensitivities