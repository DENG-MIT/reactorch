#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Weiqi Ji"
__copyright__ = "Copyright 2020, DENG"

__version__ = "0.1"
__email__ = "weiqiji@mit.edu"
__status__ = "Development"

import cantera as ct
import torch
from ruamel.yaml import YAML
from torch import nn

torch.set_default_tensor_type("torch.DoubleTensor")


class Solution(nn.Module):
    from .import_kinetics import set_nasa
    from .import_kinetics import set_reactions

    from .kinetics import forward_rate_constants_func
    from .kinetics import forward_rate_constants_func_vec
    from .kinetics import equilibrium_constants_func
    from .kinetics import reverse_rate_constants_func
    from .kinetics import wdot_func

    def __init__(self, mech_yaml=None, device=None, vectorize=False):
        super(Solution, self).__init__()

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device

        # whether the computation of reaction rate of type4 will be vectorized
        self.vectorize = vectorize

        self.gas = ct.Solution(mech_yaml)

        self.R = ct.gas_constant

        self.P_atm = torch.Tensor([ct.one_atm]).to(self.device)

        self.n_species = self.gas.n_species

        self.n_reactions = self.gas.n_reactions

        self.uq_A = nn.Parameter(torch.Tensor(self.n_reactions).fill_(1.0).to(self.device))

        self.molecular_weights = torch.Tensor([self.gas.molecular_weights]).T.to(self.device)

        with open(mech_yaml, 'r') as stream:

            yaml = YAML()

            model_yaml = yaml.load(stream)

            self.model_yaml = model_yaml

            self.set_nasa()

            self.set_reactions()

    def set_pressure(self, P):
        self.P_ref = torch.Tensor([P]).to(self.device)

    def set_states(self, TPY):

        self.T = torch.clamp(TPY[:, 0:1], min=200, max=None)

        self.logT = torch.log(self.T)

        if TPY.shape[1] == self.n_species + 2:
            self.P = TPY[:, 1:2]
            self.Y = torch.clamp(TPY[:, 2:], min=0, max=None)

        if TPY.shape[1] == self.n_species + 1:
            self.P = torch.ones_like(self.T) * self.P_ref
            self.Y = torch.clamp(TPY[:, 1:], min=0.0, max=None)

        self.Y = (self.Y.T / self.Y.sum(dim=1)).T

        self.mean_molecular_weight = 1 / torch.mm(self.Y, 1 / self.molecular_weights)

        self.density_mass = self.P / self.R / self.T * self.mean_molecular_weight

        self.Y2X()
        self.Y2C()

        self.cp_mole_func()
        self.cp_mass_func()

        self.enthalpy_mole_func()
        self.enthalpy_mass_func()

        self.entropy_mole_func()
        self.entropy_mass_func()

        # concentration of M in three-body reaction (type 2)
        self.C_M = torch.mm(self.C, self.efficiencies_coeffs)

        self.identity_mat = torch.ones_like(self.C_M)

        # for batch computation
        self.C_M2 = (self.C_M * self.is_three_body +
                     self.identity_mat * (1 - self.is_three_body))

        if self.vectorize is True:
            # for type 4
            self.C_M_type4 = torch.mm(self.C, self.efficiencies_coeffs_type4)
            self.forward_rate_constants_func_vec()

        else:

            self.forward_rate_constants_func()

        self.equilibrium_constants_func()
        self.reverse_rate_constants_func()

        self.wdot_func()

    def update_transport(self):

        self.viscosities_func()

        self.thermal_conductivity_func()

        self.binary_diff_coeffs_func()

    def set_transport(self, species_viscosities_poly, thermal_conductivity_poly,
                      binary_diff_coeffs_poly):
        # Transport Properties

        self.species_viscosities_poly = torch.from_numpy(species_viscosities_poly).to(self.device)

        self.thermal_conductivity_poly = torch.from_numpy(thermal_conductivity_poly).to(self.device)

        self.binary_diff_coeffs_poly = torch.from_numpy(binary_diff_coeffs_poly).to(self.device)

        self.poly_order = self.species_viscosities_poly.shape[0]

    def viscosities_func(self):

        self.trans_T = torch.cat([self.logT ** i for i in reversed(range(self.poly_order))], dim=1)

        self.species_viscosities = torch.mm(self.trans_T, self.species_viscosities_poly)

        self.Wk_over_Wj = torch.mm(self.molecular_weights, 1 / self.molecular_weights.T)

        self.Wj_over_Wk = 1 / self.Wk_over_Wj

        self.etak_over_etaj = torch.bmm(
            self.species_viscosities.unsqueeze(-1),
            1 / self.species_viscosities.unsqueeze(-1).view(-1, 1, self.n_species))

        self.PHI = (1 / 2.8284271247461903 / torch.sqrt(1 + self.Wk_over_Wj) *
                    (1 + self.etak_over_etaj ** 0.5 * self.Wj_over_Wk ** 0.25) ** 2)

        self.viscosities = (self.X.clone() * self.species_viscosities /
                            torch.bmm(self.PHI, self.X.unsqueeze(-1)).squeeze(-1)).sum(dim=1,
                                                                                       keepdim=True)

    def thermal_conductivity_func(self):

        self.species_thermal_conductivity = torch.mm(self.trans_T, self.thermal_conductivity_poly)

        self.thermal_conductivity = 0.5 * (
                (self.X.clone() * self.species_thermal_conductivity).sum(dim=1, keepdim=True) +
                1 / (self.X.clone() / self.species_thermal_conductivity).sum(dim=1, keepdim=True)
        )

    def binary_diff_coeffs_func(self):

        self.binary_diff_coeffs = torch.mm(
            self.trans_T, self.binary_diff_coeffs_poly).view(-1, self.n_species, self.n_species)

        self.X_eps = self.X.clamp_(1e-12)

        self.XjWj = (torch.mm(self.X_eps, self.molecular_weights) -
                     self.X_eps * self.molecular_weights.T)

        self.XjDjk = torch.bmm(self.X_eps.view(-1, 1, self.n_species), 1 / self.binary_diff_coeffs).squeeze(
            1) - self.X_eps / self.binary_diff_coeffs.diagonal(dim1=-2, dim2=-1)

        self.mix_diff_coeffs = self.XjWj / self.XjDjk / self.mean_molecular_weight / self.P * self.P_atm

    # Magic Functions

    def C2X(self):
        self.X = (self.C.T / self.C.sum(dim=1)).T

    def Y2X(self):
        self.X = self.Y * self.mean_molecular_weight / self.molecular_weights.T

    def Y2C(self):
        self.C = self.Y * self.density_mass / self.molecular_weights.T

    def X2C(self):
        self.C = self.X * self.density_mass / self.mean_molecular_weight

    def X2Y(self):
        self.Y = self.X * self.molecular_weights.T / self.mean_molecular_weight

    def cp_mole_func(self):

        self.cp_T = torch.cat([self.T ** 0, self.T, self.T ** 2, self.T ** 3, self.T ** 4], dim=1)

        self.cp = (torch.mm(self.cp_T, self.nasa_low[:, :5].T) * (self.T <= 1000).int() +
                   torch.mm(self.cp_T, self.nasa_high[:, :5].T) * (self.T > 1000).int())

        self.cp_mole = (self.R * self.cp * self.X.clone()).sum(dim=1)

    def cp_mass_func(self):

        self.cp_mass = self.cp_mole / self.mean_molecular_weight.T

    def enthalpy_mole_func(self):

        self.H_T = torch.cat((self.T ** 0, self.T / 2, self.T ** 2 / 3,
                              self.T ** 3 / 4, self.T ** 4 / 5, 1 / self.T), dim=1)

        self.H = (torch.mm(self.H_T, self.nasa_low[:, :6].T) * (self.T <= 1000).int() +
                  torch.mm(self.H_T, self.nasa_high[:, :6].T) * (self.T > 1000).int())

        self.H = self.H * self.R * self.T

        self.partial_molar_enthalpies = self.H

        self.enthalpy_mole = (self.H * self.X).sum(dim=1)

    def enthalpy_mass_func(self):

        self.h = self.H / self.molecular_weights.T

        self.enthalpy_mass = (self.h * self.Y).sum(dim=1)

    def entropy_mole_func(self):

        self.S_T = torch.cat((torch.log(self.T), self.T, self.T ** 2 / 2,
                              self.T ** 3 / 3, self.T ** 4 / 4, self.T ** 0), dim=1)

        self.S0 = (
                torch.mm(self.S_T, self.nasa_low[:, [0, 1, 2, 3, 4, 6]].T) * (self.T <= 1000).int() +
                torch.mm(self.S_T, self.nasa_high[:, [0, 1, 2, 3, 4, 6]].T) * (self.T > 1000).int())

        self.S0 = self.S0 * self.R

        self.S = self.S0 - self.R * torch.log(self.X) - self.R * torch.log(self.P / ct.one_atm)

        self.partial_molar_entropies = self.S

        self.entropy_mole = (self.S * self.X).sum(dim=1)

    def entropy_mass_func(self):

        self.s = self.S / self.molecular_weights.T

        self.entropy_mass = (self.s * self.Y).sum(dim=1)

    def Ydot_func(self):

        self.Ydot = self.wdot / self.density_mass * self.molecular_weights.T

    def Xdot_func(self):

        self.Xdot = self.Ydot * self.mean_molecular_weight / self.molecular_weights.T

    def Tdot_func(self):

        self.Tdot = -((self.partial_molar_enthalpies * self.wdot).sum(dim=1) /
                      self.density_mass.T / self.cp_mass).T

    def TXdot_func(self):

        self.Tdot_func()

        self.Xdot_func()

        self.TXdot = torch.cat((self.Tdot, self.Xdot), dim=1)

        return self.TXdot

    def TYdot_func(self):

        self.Tdot_func()

        self.Ydot_func()

        self.TYdot = torch.cat((self.Tdot, self.Ydot), dim=1)

        return self.TYdot
