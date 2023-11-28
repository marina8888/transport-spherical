import pandas as pd
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12})

# main code for diffusion flux calculation + plot


class TransportCalc:
    """
    Base class to import a flame from csv to conduct diffusion calculations. (Avoids rerunning flame calculation over and over).
    Contains two dataframes - species full (whole domain) and inputs (all parameters and species)
    """

    def __init__(self, mechanism_path: str, species: str):
        """
        Main call for diffusion calculations (multicomponent formulation)
        :param mechanism_path:
        :param species: select species for diffusion flux calculations
        """
        # Prepare to create a flame object:
        self.mech = mechanism_path
        self.gas = ct.Solution(self.mech, transport="Multi")
        self.species = self.gas.species_index(species)
        self.flamelet()  # modify this function based on what kind of flame you want to run - self.flamelet_liu() was the test data
        self.collect_data()
        self.calculate_transport()

    def flamelet(self):
        """
        Run impinging jet flamelet. Change flame properties here.
        :return:
        """
        # just in case, reinitialise gas since its passed around internally:
        self.gas = ct.Solution(self.mech)

        self.gas.set_equivalence_ratio(1.2, {"NH3": 0.7, "H2": 0.3}, {"O2": 1.0, "N2": 3.76})
        self.gas.TP = 298, ct.one_atm
        self.ratio = 3
        self.slope = 0.1
        self.curve = 0.3
        self.prune = 0

        axial_velocity = 0.405038 #0.24747
        mass_flux = self.gas.density * axial_velocity  # units kg/m2/s

        # Domain width of 2 cm:
        width = 0.02
        self.grid = np.linspace(0, 0.02, 500)
        # select flame type to solve:
        self.f = ct.ImpingingJet(gas=self.gas, grid=self.grid)
        self.f.set_initial_guess("equil")  # assume adiabatic equilibrium products

        # set the mass flow rate at the inlet and wall temp
        self.f.inlet.mdot = mass_flux
        self.f.surface.T = 577.05172 # 500.949

        # self.f.set_grid_min(5e-6)
        self.f.transport_model = "Multi"
        self.f.soret_enabled = True

        self.f.set_max_time_step(1000)
        self.f.set_max_grid_points(domain=1, npmax=50000)
        self.f.set_refine_criteria(
            ratio=self.ratio, slope=self.slope, curve=self.curve, prune=self.prune
        )
        self.f.solve(loglevel=1, auto=True)

    def flamelet_fillo(self):
        """
        Run same flame configuration as liu paper for comparison
        http://dx.doi.org/10.1016/j.ijhydene.2015.04.133
        :return:
        """
        # just in case, reinitialise gas since its passsed around internally:
        self.gas = ct.Solution(self.mech, transport_model="Multi")

        self.gas.set_equivalence_ratio(0.4, {"H2": 1.0}, {"O2": 1.0, "N2": 3.76})
        self.gas.TP = 298, ct.one_atm
        self.ratio = 2.0
        self.slope = 0.1
        self.curve = 0.05
        self.prune = 0.002

        axial_velocity = 0.002
        mass_flux = self.gas.density * axial_velocity  # units kg/m2/s

        # Domain width of 12 cm for non uniform grid, or for uniform grid, set spacing:
        self.width = 0.02
        self.grid = np.linspace(0, 0.02, 100)
        # select flame type to solve:
        self.f = ct.FreeFlame(gas=self.gas, grid=self.grid)
        # self.f.set_initial_guess()  # assume adiabatic equilibrium products

        # set the mass flow rate at the inlet and wall temp
        self.f.inlet.mdot = mass_flux

        self.f.transport_model = "Multi"
        self.f.soret_enabled = True

        # self.f.set_max_time_step(1000)
        self.f.set_max_grid_points(domain=1, npmax=20000)
        self.f.set_refine_criteria(
            ratio=self.ratio, slope=self.slope, curve=self.curve, prune=self.prune
        )
        self.f.solve(loglevel=0, auto=True)

    def flamelet_liu(self):
        """
        Run same flame configuration as liu paper for comparison
        http://dx.doi.org/10.1016/j.ijhydene.2015.04.133
        :return:
        """
        # just in case, reinitialise gas since its passsed around internally:
        self.gas = ct.Solution(self.mech, transport_model="Multi")

        self.gas.set_equivalence_ratio(
            1.0, {"H2": 0.4, "CO2": 0.3, "CH4": 0.3}, {"O2": 1.0, "N2": 3.76}
        )
        self.gas.TP = 298, ct.one_atm
        self.ratio = 2.0
        self.slope = 0.1
        self.curve = 0.05
        self.prune = 0.002

        axial_velocity = 0.002
        mass_flux = self.gas.density * axial_velocity  # units kg/m2/s

        # Domain width of 12 cm for non uniform grid, or for uniform grid, set spacing:
        self.width = 0.02
        self.grid = np.linspace(0, 0.02, 100)
        # select flame type to solve:
        self.f = ct.FreeFlame(gas=self.gas, grid=self.grid)
        # self.f.set_initial_guess()  # assume adiabatic equilibrium products

        # set the mass flow rate at the inlet and wall temp
        self.f.inlet.mdot = mass_flux

        self.f.transport_model = "Multi"
        self.f.soret_enabled = False

        # self.f.set_max_time_step(1000)
        self.f.set_max_grid_points(domain=1, npmax=20000)
        self.f.set_refine_criteria(
            ratio=self.ratio, slope=self.slope, curve=self.curve, prune=self.prune
        )
        self.f.solve(loglevel=0, auto=True)

    def derivative(self, x, y):
        """
        Generic derivative function
        :param y: self.f.value
        :param x: self.f.value
        :return:
        """
        dydx = np.zeros(y.shape, y.dtype.type)

        dx = np.diff(x)
        dy = np.diff(y)
        dydx[0:-1] = dy / dx
        dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])

        return dydx

    def collect_data(self):
        """
        Function to collect key values from flame object needed for the equation.
        :return:
        """
        self.gas_temp = ct.Solution(self.mech, transport_model="Multi")
        self.mean_mol_w = np.zeros(self.f.T.shape)
        self.dT_dx = np.zeros(self.f.T.shape)
        self.dX_dx = np.zeros(self.f.Y.shape)
        self.dY_dx = np.zeros(self.f.Y.shape)
        self.u_c = np.zeros(self.f.T.shape)
        self.D_kT = np.zeros(self.f.T.shape)
        self.D_kj = np.zeros(self.f.Y.shape)
        self.D_kM = np.zeros(self.f.T.shape)

        # fill mean molecular weight, and diffusion flux values:
        for n in range(len(self.f.grid)):
            self.gas_temp.TPY = self.f.T[n], self.f.P, self.f.Y[:, n]
            self.mean_mol_w[n] = self.gas_temp.mean_molecular_weight
            self.D_kT[n] = self.gas_temp.thermal_diff_coeffs[
                self.species
            ]  # we only care about species k
            self.D_kj[:, n] = self.gas_temp.multi_diff_coeffs[
                :, self.species
            ]  # we only care about species k (col) in gas all (row)
            self.D_kM[n] = self.gas_temp.mix_diff_coeffs[self.species]

        # fill by species - derivative values and molecular diffusion flux
        self.dT_dx = self.derivative(self.f.grid, self.f.T)
        self.ln_dT_dx = self.derivative(self.f.grid, np.log(self.f.T))

        for s in range(len(self.gas_temp.species_names)):
            self.dX_dx[s, :] = self.derivative(self.f.grid, self.f.X[s, :])
            self.dY_dx[s, :] = self.derivative(self.f.grid, self.f.Y[s, :])

    def calculate_transport(self):
        # thermal flux calculation:
        thermal_flux = -1 * self.D_kT * self.ln_dT_dx

        # mixture averaged flux calculation:
        # mixav_flux = -1 * (self.f.density * self.D_kM * self.dY_dx[self.species]) + (self.f.density * self.f.Y[self.species] * self.u_c)
        mixav_flux = -1 * self.f.density * self.D_kM * self.dY_dx[self.species]

        # multicomponent flux summary
        molecular_flux_sum = np.zeros(self.f.Y.shape)
        for s in range(len(self.gas.species_names)):
            if s != self.species:
                molecular_flux_sum[s, :] = self.gas_temp.molecular_weights[s] * np.multiply(self.D_kj[s, :], self.dX_dx[s, :])

        # NASA paper version of equation, to match thermal flux:
        multic_flux = 1 * np.divide(
            (
                np.sum(molecular_flux_sum, axis=0)
                * self.f.density
                * self.gas_temp.molecular_weights[self.species]
            ),
            (self.mean_mol_w**2),
        )

        # plt.plot(self.f.T, thermal_flux, label = 'thermal flux')
        # plt.plot(self.f.T, multic_flux, label ='multicomponent molecular flux')
        # plt.plot(self.f.T, mixav_flux, label='mixav molecular flux')
        # plt.ylabel('flux, kg/m^2*s')
        # plt.xlabel('temperature, K')
        # plt.legend()
        # plt.title(f' diffusion of {self.gas.species_name(self.species)}')
        # plt.show()
        # plt.rcParams.update({'font.size': 16})

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(self.f.grid, thermal_flux, color="blue",linewidth=3, label="thermal flux")
        ax1.plot(self.f.grid, multic_flux, color="green",linewidth=3,label="ordinary (MultiC) flux")
        ax1.plot(self.f.grid, mixav_flux, color="red",linewidth = 3, label ='ordinary (MixAv) flux')
        ax2.plot(self.f.grid, self.f.T, linewidth=3, color="magenta", label="Temperature")
        ax2.plot(self.f.grid, self.f.X[self.species]*5000, linewidth=3, color="black",label="Species conc. (mol)")
        ax1.set_ylabel(rf"flux, $kg/m^2s $ (x 100)", fontsize=16)
        sp = self.gas.species_name(self.species)
        ax2.set_ylabel(rf"Temperature, K or $X_{{{sp}}}$ (x 5000)", fontsize=16)
        ax1.set_xlabel("grid, m", fontsize=16)
        ax1.set_xlim(0.007, 0.02)
        ax2.set_ylim(0, 3000)
        ax1.set_ylim(-0.02, 0.02)
        # ax2.legend(fontsize=12)
        # ax1.legend(fontsize=12)
        ax1.locator_params(axis="x", nbins=5)
        plt.title(f" diffusion flux of {self.gas.species_name(self.species)}", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"rich_gotama_{self.gas.species_name(self.species)}_diff.jpg")
        plt.show()


class TransportCalcSheet(TransportCalc):
    """
    Base class to import a flame from csv to conduct diffusion calculations. (Avoids rerunning flame calculation over and over).
    Contains two dataframes - species full (whole domain) and inputs (all parameters and species)
    """

    def __init__(
        self, mechanism_path: str, species: str, filename: str, number_nonspec
    ):
        """

        :param mechanism_path:
        :param species:
        :param filename:
        :param number_nonspec: number of columns which are not species (assuming all -1 are stacked at the front of the sheet)
        """
        # Prepare to create a flame object:
        self.filename = filename
        self.num_nonspec = number_nonspec
        super().__init__(mechanism_path, species)
        self.mech = mechanism_path
        self.gas = ct.Solution(self.mech, transport="Multi")
        self.species = self.gas.species_index(species)
        self.flamelet()  # modify this function based on what kind of flame you want to run - self.flamelet_liu() was the test data

    def flamelet(self):
        self.f = pd.read_excel(self.filename)
        print(self.f)

    def collect_data(self):
        """
        Function to collect key values from flame object needed for the equation.
        :return:
        """
        self.gas_temp = ct.Solution(self.mech, transport_model="Multi")

        self.mean_mol_w = np.zeros(self.f.shape[0])
        self.dT_dx = np.zeros(self.f.shape[0])
        self.dX_dx = np.zeros((self.f.shape[0], self.f.shape[1] - self.num_nonspec))
        self.dY_dx = np.zeros((self.f.shape[0], self.f.shape[1] - self.num_nonspec))
        self.u_c = np.zeros(self.f.shape[0])
        self.D_kT = np.zeros(self.f.shape[0])
        self.D_kj = np.zeros((self.f.shape[0], self.f.shape[1] - self.num_nonspec))
        self.D_kM = np.zeros(self.f.shape[0])

        self.f = self.f.rename(columns={"T (K)": "T", "z (m)": "grid"})
        self.f.P = ct.one_atm
        self.Y = self.f.iloc[:, self.num_nonspec - 1 : -1]
        print(self.Y[:, 5])
        # fill mean molecular weight, and diffusion flux values:
        for n in range(len(self.f.grid)):
            self.gas_temp.TPY = self.f["T"][n], self.f.P, self.Y[:, n]
            self.mean_mol_w[n] = self.gas_temp.mean_molecular_weight
            self.D_kT[n] = self.gas_temp.thermal_diff_coeffs[self.species]
            self.D_kj[:, n] = self.gas_temp.multi_diff_coeffs[
                :, self.species
            ]  # we only care about species k (col) in gas all (row)
            self.D_kM[n] = self.gas_temp.mix_diff_coeffs[self.species]

        # fill by species - derivative values and molecular diffusion flux
        self.dT_dx = self.derivative(self.f.grid, self.f.T)
        self.ln_dT_dx = self.derivative(self.f.grid, np.log(self.f.T))

        for s in range(len(self.gas_temp.species_names)):
            self.dX_dx[s, :] = self.derivative(self.f.grid, self.f.X[s, :])
            self.dY_dx[s, :] = self.derivative(self.f.grid, self.f.Y[s, :])

    def calculate_transport(self):
        pass