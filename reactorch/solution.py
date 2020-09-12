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
    from .import_kinetics import set_transport

    from .kinetics import forward_rate_constants_func
    from .kinetics import forward_rate_constants_func_vec
    from .kinetics import equilibrium_constants_func
    from .kinetics import reverse_rate_constants_func
    from .kinetics import wdot_func
    from .kinetics import Ydot_func
    from .kinetics import Xdot_func
    from .kinetics import Tdot_func
    from .kinetics import TXdot_func
    from .kinetics import TYdot_func

    from .thermo import cp_mole_func
    from .thermo import cp_mass_func
    from .thermo import enthalpy_mole_func
    from .thermo import enthalpy_mass_func
    from .thermo import entropy_mole_func
    from .thermo import entropy_mass_func

    from .transport import update_transport
    from .transport import viscosities_func
    from .transport import thermal_conductivity_func
    from .transport import binary_diff_coeffs_func

    from .magic_function import C2X, Y2X, Y2C, X2C, X2Y

    def __init__(self, mech_yaml=None, device=None, vectorize=False,
                 is_clip=True, is_norm=True, is_wdot_vec=True):
        super(Solution, self).__init__()

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device

        # whether the computation of reaction rate of type 4 will be vectorized
        self.vectorize = vectorize
        self.is_clip = is_clip
        self.is_norm = is_norm
        self.is_wdot_vec = is_wdot_vec

        self.gas = ct.Solution(mech_yaml)

        self.R = ct.gas_constant

        self.one_atm = torch.Tensor([ct.one_atm]).to(self.device)

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

    def set_states(self, TPY, eval_rate=True):

        self.T = torch.clamp(TPY[:, 0:1], min=200, max=None)

        self.logT = torch.log(self.T)

        if TPY.shape[1] == self.n_species + 2:
            self.P = TPY[:, 1:2]
            if self.is_clip:
                self.Y = torch.clamp(TPY[:, 2:], min=0, max=None)
            else:
                self.Y = TPY[:, 2:]

        if TPY.shape[1] == self.n_species + 1:
            self.P = torch.ones_like(self.T) * self.P_ref
            if self.is_clip:
                self.Y = torch.clamp(TPY[:, 1:], min=0.0, max=None)
            else:
                self.Y = TPY[:, 1:]

        if self.is_norm:
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

        if eval_rate:

            # concentration of M in three-body reaction (type 2)
            self.C_M = torch.mm(self.C, self.efficiencies_coeffs)

            self.identity_mat = torch.ones_like(self.C_M)

            # for batch computation
            self.C_M2 = (self.C_M * self.is_three_body +
                         self.identity_mat * (1 - self.is_three_body))

            if self.vectorize:
                # for reaction of type 4
                self.C_M_type4 = torch.mm(self.C, self.efficiencies_coeffs_type4)
                self.forward_rate_constants_func_vec()

            else:

                self.forward_rate_constants_func()

            self.equilibrium_constants_func()
            self.reverse_rate_constants_func()

            self.wdot_func()
