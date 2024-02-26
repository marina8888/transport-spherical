import cantera as ct
import pandas as pd
import numpy as np
from src.settings.logger import LogConfig
import os
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

from src.calculations.rop_sens import BaseFlame
import src.settings.config_loader as config

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
        self.gas = ct.Solution(f"{config.INPUT_DIR_MECH}/{self.mech_name}.yaml")
        self.gas.TP = self.TP
        self.gas.set_equivalence_ratio(self.phi, fuel=self.fuel, oxidizer=self.oxidizer)

    def configure_flame(self):
        # we are using an ImpingingJet class but there are others that might be suitable for other experiments
        self.f = ct.ImpingingJet(gas=self.gas, width=config.STAGNATION_FLAME_WIDTH)
        self.f.set_max_grid_points(domain=1, npmax=config.MAX_GRID)
        self.f.inlet.mdot = self.vel * self.gas.density
        self.f.surface.T = self.T
        self.f.transport_model = "Multi"
        self.f.soret_enabled = True
        self.f.radiation_enabled = False
        self.f.set_initial_guess("equil")

        # if os.path.isfile(f"{config.OUTPUT_DIR_NUMERICAL}/init.csv"):
        #     self.f.set_initial_guess(f"{config.OUTPUT_DIR_NUMERICAL}/init.csv")  # assume adiabatic equilibrium products
        #     print("starting from previous solution saved as 'init.csv")
        # else:

        self.f.set_refine_criteria(ratio=3, slope=config.SLOPE, curve=config.CURVE, prune=config.PRUNE)

    def check_solution_file_exists(self, filename, columns):
        if not (os.path.exists(filename)):
            pd.DataFrame(columns=columns).to_csv(f"{filename}")

    def solve(self):
        try:
            self.f.solve(loglevel=1, auto=True)
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
                    "pos_temp": self.pos_temp(),
                    "max_temp": self.max_temp()
                }
                data_y = dict(zip(self.gas.species_names, self.f.X[:, -1]))
                data = {**data_x, **data_y}
                df = pd.json_normalize(data)
                filename = f"{config.OUTPUT_DIR_NUMERICAL}/{self.blend}_{self.mech_name}.csv"
                self.check_solution_file_exists(filename, df.columns)
                df.to_csv(f"{filename}", mode="a", header=False)
        except ct.CanteraError as e:
            self.logger.info(f"simulation run error: {e}")
            pass

    def solve_domain(self):
        try:
            self.f.solve(loglevel=0, auto=True)
            plt.plot(self.f.grid, self.f.T)
            plt.show()
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
                    "pos_temp": [self.pos_temp()] * len(self.f.grid),
                    "max_temp": [self.max_temp()] * len(self.f.grid),
                }
                data_y = dict(zip(self.gas.species_names, self.f.X))
                data = {**data_x, **data_y}
                df = pd.DataFrame(data)
                filename = f"{self.blend}_{self.mech_name}.csv"

                self.check_solution_file_exists(filename, df.columns)
                df.to_csv(f"{filename}", mode="a", header=False)
        except ct.CanteraError:
            pass


    def pos_temp(self):
        """
        Calculate the strain rate for the flame studied
        :param grid:
        :param velocity:
        :return:
        """
        max_temp_position = np.unravel_index(np.argmax(self.f.T), self.f.T.shape)

        # Assuming grid_point() returns the coordinates of a grid point
        max_temp_grid_point = self.f.grid[max_temp_position]

        return max_temp_grid_point

    def max_temp(self):
        """
        Calculate the strain rate for the flame studied
        :param grid:
        :param velocity:
        :return:
        """
        print(self.f.T.max())
        return self.f.T.max()