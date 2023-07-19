import cantera as ct
import pandas as pd
import numpy as np
from src.settings.logger import LogConfig
import os
from scipy.signal import argrelextrema

from src.calculations.rop_sens import BaseFlame
from src.settings.filepaths import mech_dir

# this file runs a stagnation stabilised flame in Cantera

class StagnationFlame(BaseFlame):
    def __init__(self, oxidizer, blend, fuel, phi, T_in, P, T, vel, mech_name, species = None):
        self.oxidizer = str(oxidizer)
        self.blend = blend
        self.fuel = fuel
        self.phi = phi
        self.T_in = T_in
        self.P = P
        self.TP = (T_in, P)
        self.T = T
        self.vel = vel
        self.species = species
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
        self.f.set_max_grid_points(domain=1, npmax=1400)
        self.f.inlet.mdot = self.vel * self.gas.density
        self.f.surface.T = self.T
        self.f.transport_model = "Multi"
        self.f.soret_enabled = True
        self.f.radiation_enabled = False
        self.f.set_initial_guess("equil")  # assume adiabatic equilibrium products
        self.f.set_refine_criteria(ratio=3, slope=0.02, curve=0.04, prune=0.0001)
        # self.f.set_refine_criteria(ratio=3, slope=0.012, curve=0.028, prune=0.0001)
        # self.f.set_refine_criteria(ratio=3, slope=0.2, curve=0.4, prune=0)

    def check_solution_file_exists(self, filename, columns):
        if not (os.path.exists(filename)):
            pd.DataFrame(columns=columns).to_csv(f"{filename}")


    def solve(self):
        try:
            self.f.solve(loglevel=0, auto=True)
            if max(self.f.T) < float(self.T)+100:
                self.logger.info(f"\n FLAME AT phi = {self.phi} NOT IGNITED!")
                return 0
            else:
                self.logger.info(f"\n FLAME AT phi = {self.phi}  IGNITED!")
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
                    "strain": self.strain,
                }
                data_y = dict(zip(self.gas.species_names, self.f.X[:, -1]))
                data = {**data_x, **data_y}
                df = pd.json_normalize(data)
                filename = f"{self.blend}_{self.mech_name}.csv"

                self.check_solution_file_exists(filename, df.columns)
                df.to_csv(f"{filename}", mode="a", header=False)

        except ct.CanteraError:
            pass

    def solve_domain(self):
        try:
            self.f.solve(loglevel=1, auto=True)
            if max(self.f.T) < float(self.T)+100:
                self.logger.info(f"\n FLAME AT phi = {self.phi} NOT IGNITED!")
                return 0
            else:
                self.logger.info(f"\n FLAME AT phi = {self.phi}  IGNITED!")
                data_x = {
                    "oxidizer": [self.oxidizer] * len(self.f.grid),
                    "phi": [self.phi] * len(self.f.grid),
                    "grid": self.f.grid,
                    "T_in": [self.T_in] * len(self.f.grid),
                    "T": self.f.T,
                    "T_f": [self.T] * len(self.f.grid),
                    "P": [self.f.P] * len(self.f.grid),
                    "vel": [self.vel] * len(self.f.grid),
                    "blend": [self.blend] * len(self.f.grid),
                    "fuel": [self.fuel] * len(self.f.grid),
                    "strain_t1": [self.strain_t1] * len(self.f.grid),
                    "strain_t2": [self.strain_t2] * len(self.f.grid),
                }
                data_y = dict(zip(self.gas.species_names, self.f.X))
                data = {**data_x, **data_y}
                df = pd.DataFrame(data)
                filename = f"{self.blend}_{self.mech_name}.csv"
                self.check_solution_file_exists(filename, df.columns)
                df.reset_index(drop=True, inplace=True)
                print(df)
                df.to_csv(f"{filename}", sep = ',', index=False)

        except ct.CanteraError:
            pass


    def strain_t1(self):
        """
        Calculate the strain rate for the flame studied
        :param grid:
        :param velocity:
        :return:
        """
        strain = 0

        minima_locs = argrelextrema(self.f.velocity, np.less_equal)
        grad_vel = np.gradient(self.f.velocity, self.f.grid)

        try:
            idx = int([tup[0] for tup in minima_locs][0])
            strain = np.min(grad_vel[0:idx])
        except TypeError:
            print("cannot do strain, strain = 0")
        return -1 * strain

    def strain_t2(self):
        """
        Calculate the strain rate for the flame studied
        :param grid:
        :param velocity:
        :return:
        """

        return 2*self.vel/0.02