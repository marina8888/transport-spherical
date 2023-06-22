import cantera as ct
from src.settings.filepaths import mech_path, output_dir
import pandas as pd
from src.settings.logger import LogConfig
import os

# this file runs a stagnation stabilised flame

class StagnationFlame:
    def __init__(self, oxidizer, blend, fuel, phi, T_in, P, T, vel, flash_point, mech_name):
        self.oxidizer = oxidizer
        self.blend = blend
        self.fuel = str(fuel)
        self.phi = phi
        self.T_in = T_in
        self.P = P
        self.TP = (T_in, P)
        self.T = T
        self.vel = vel
        self.flash_point = flash_point
        self.mech_name = mech_name
        self.logger = LogConfig.configure_logger(__name__)

    def configure_gas(self):
        # gas is a solution class from the Cantera library
        self.gas = ct.Solution(f"{mech_path}/{self.mech_name}.cti")
        self.gas.TP = self.TP
        self.gas.set_equivalence_ratio(self.phi, fuel=self.fuel, oxidizer=self.oxidizer)

    def configure_flame(self):
        # we are using an ImpingingJet class but there are others that might be suitable for other experiments
        self.f = ct.ImpingingJet(gas=self.gas, width=0.02)
        self.f.set_max_grid_points(domain=1, npmax=1800)
        self.f.inlet.mdot = self.vel * self.gas.density
        self.f.surface.T = self.T
        self.f.transport_model = "Multi"
        self.f.soret_enabled = True
        self.f.radiation_enabled = False
        self.f.set_initial_guess("equil")  # assume adiabatic equilibrium products
        self.f.set_refine_criteria(ratio=3, slope=0.3, curve=0.7 , prune=0)

    def check_solution_file_exists(self, filename, columns):
        if not (os.path.exists(filename)):
            pd.DataFrame(columns=columns).to_csv(f"{filename}")

    def solve(self):
        try:
            self.f.solve(loglevel=0, auto=True)
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
                    "blend": self.blend_H2,
                    "fuel": self.fuel,
                    "oxidizer": self.oxidizer,
                }
                data_y = dict(zip(self.gas.species_names, self.f.X[:, -1]))
                data = {**data_x, **data_y}
                df = pd.json_normalize(data)
                filename = f"{output_dir}/{self.blend_H2}_{self.mech_name}.csv"

                self.check_solution_file_exists(filename, df.columns)
                df.to_csv(f"{filename}", mode="a", header=False)

        except ct.CanteraError:
            pass