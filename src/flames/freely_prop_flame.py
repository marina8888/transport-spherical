import cantera as ct
import pandas as pd
import os

from src.settings.filepaths import mech_dir, output_dir_numerical_output, output_dir_numerical_domain
from src.settings.logger import LogConfig
from src.calculations.rop_sens import BaseFlame
# this file runs a freely propagating flame model in Cantera
class FreelyPropFlame(BaseFlame):
    def __init__(self, oxidizer, blend, fuel, phi, T_in, P, mech_name, species = None):
        self.oxidizer = str(oxidizer)
        self.blend = blend
        self.fuel = str(fuel)
        self.phi = phi
        self.T_in = T_in
        self.P = P
        self.TP = (T_in, P)
        self.mech_name = mech_name
        self.species = species
        self.logger = LogConfig.configure_logger(__name__)

    def configure_gas(self):
        # gas is a solution class from the Cantera library
        self.gas = ct.Solution(f"{mech_dir}/{self.mech_name}.yaml")
        self.gas.TP = self.TP
        self.gas.set_equivalence_ratio(self.phi, fuel=self.fuel, oxidizer=self.oxidizer)

    def configure_flame(self):
        # we are using an ImpingingJet class but there are others that might be suitable for other experiments
        self.f = ct.FreeFlame(gas=self.gas, width=0.1)
        self.f.set_max_grid_points(domain=1, npmax=2500)
        self.f.transport_model = "Multi"
        self.f.soret_enabled = True
        self.f.radiation_enabled = False
        self.f.set_refine_criteria(ratio=3, slope=0.012, curve=0.024, prune=0)

    def check_solution_file_exists(self, filename, columns):
        if not (os.path.exists(filename)):
            pd.DataFrame(columns=columns).to_csv(f"{filename}")

    def solve(self):
        try:
            self.f.solve(loglevel=1, auto=True)
            if max(self.f.T) < self.T_in+100:
                return 0
            else:
                data_x = {
                    "phi": self.phi,
                    "grid": len(self.f.grid),
                    "T_in": self.T_in,
                    "P": self.P,
                    "blend": self.blend,
                    "fuel": self.fuel,
                    "oxidizer": self.oxidizer,
                    "T_max": max(self.f.T),
                }
                data_y = {"flame_speed": self.f.velocity[0]}
                data = {**data_x, **data_y}
                df = pd.json_normalize(data)
                filename = f"{output_dir_numerical_output}/{self.blend}_{self.mech_name}.csv"

                self.check_solution_file_exists(filename, df.columns)
                df.to_csv(f"{filename}", mode="a", header=False)

        except ct.CanteraError:
            pass

    def solve_domain(self):
        try:
            self.f.solve(loglevel=0, auto=True)
            if max(self.f.T) < self.T_in+100:
                return 0
            else:
                data_x = {
                    "phi": self.phi,
                    "grid": len(self.f.grid),
                    "T_in": self.T_in,
                    "P": self.P,
                    "blend": self.blend,
                    "fuel": self.fuel,
                    "oxidizer": self.oxidizer,
                }
                data_y = {"flame_speed": self.f.velocity[0]}
                data_y = dict(zip(self.gas.species_names, self.f.X))
                data = {**data_x, **data_y}
                df = pd.json_normalize(data)
                filename = f"{output_dir_numerical_domain}/{self.blend}_{self.mech_name}.csv"

                # self.check_solution_file_exists(filename, df.columns)
                df.to_csv(f"{filename}", mode="a", header=False)

        except ct.CanteraError:
            pass