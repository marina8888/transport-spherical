# def get_sens_thermo_all(self):
#     """
#     Sensitivity of parameter of interest due to change in entropy.
#     @return:
#     """
#     """
#     THIS IS INCORRECT, CORRECT BEFORE USING!!
#     Sensitivity of all species to change in entropy and enthalpy
#
#     @return:
#     """
#
#     def calculate_sens_entropy(s, df):
#         def perturb(sim, i, dp):
#             S = sim.gas.species(i)
#             print(f"running sensitivity wrt to entropy of: {S}")
#             st = S.thermo
#             coeffs = st.coeffs
#             coeffs[[7, 14]] += dp * coeffs[[7, 14]] / ct.gas_constant
#             snew = ct.NasaPoly2(st.min_temp, st.max_temp, st.reference_pressure, coeffs)
#             S.thermo = snew
#             sim.gas.modify_species(sim.gas.species_index(i), S)
#             sens_vals = self.f.solve_adjoint(perturb, len(self.gas.species()), self.dgdx) / self.spec_0
#             df[s] = sens_vals
#
#         return df
#
#     def calculate_sens_enthalpy(s, df):
#         def perturb(sim, i, dp):
#             S = sim.gas.species(i)
#             print(f"running sensitivity wrt to enthalpy of: {S}")
#             st = S.thermo
#             coeffs = st.coeffs
#             coeffs[[6, 13]] += dp * coeffs[[6, 13]] / ct.gas_constant
#             snew = ct.NasaPoly2(st.min_temp, st.max_temp, st.reference_pressure, coeffs)
#             S.thermo = snew
#             sim.gas.modify_species(sim.gas.species_index(i), S)
#
#         sens_vals = self.f.solve_adjoint(perturb, len(self.gas.species()), self.dgdx) / self.spec_0
#         df[s] = sens_vals
#         return df
#
#     df_enthalpy = pd.DataFrame(index=self.gas.species())
#     df_entropy = pd.DataFrame(index=self.gas.species())
#
#     for s in self.gas.species_names:
#         print
#         self.species = s
#         self.solver_adjoint_init()
#         df_entropy = calculate_sens_entropy(s, df_entropy)
#         df_enthalpy = calculate_sens_enthalpy(s, df_enthalpy)
#     df_entropy.to_csv(
#         f"{config.OUTPUT_DIR_THERMO}/SENS_ENTROPY_{type(self).__name__}_{config.SENS_SPECIES_LOC}_{self.species}_{self.blend}_{self.phi}_{self.mech_name}.csv")
#     df_enthalpy.to_csv(
#         f"{config.OUTPUT_DIR_THERMO}/SENS_ENTHALPY_{type(self).__name__}_{config.SENS_SPECIES_LOC}_{self.species}_{self.blend}_{self.phi}_{self.mech_name}.csv")
#     return df_entropy, df_enthalpy