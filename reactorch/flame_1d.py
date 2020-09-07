#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Weiqi Ji"
__copyright__ = "Copyright 2020, DENG"

__version__ = "0.1"
__email__ = "weiqiji@mit.edu"
__status__ = "Development"

import torch
from torch import nn

torch.set_default_tensor_type("torch.DoubleTensor")


class Flame_1d(nn.Module):
    def __init__(self, sol, device=None):
        super(Flame_1d, self).__init__()

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device

        self.sol = sol

    def set_states(self, TPY):

        self.sol.set_states(TPY)

        self.update_u()
        self.update_diffusive_fluxes()
        self.update_dTdx()
        self.update_dYdx()
        self.update_rhs()

    def set_x(self, x):

        self.x = x
        self.grid = self.x

    def set_mdot(self, m_dot):

        self.m_dot = torch.Tensor([m_dot]).to(self.device)

    def update_u(self):

        self.u = self.m_dot / self.sol.density_mass

    def grad_to_x(self, output):

        if output.shape[1] == 1:

            dydt = torch.autograd.grad(outputs=output.sum(),
                                       inputs=self.x,
                                       retain_graph=True,
                                       create_graph=True,
                                       allow_unused=True)[0]

        else:
            dydt = torch.zeros_like(output)

            for i in range(output.shape[1]):

                dydt[:, i] = torch.autograd.grad(outputs=output[:, i].sum(),
                                                 inputs=self.x,
                                                 retain_graph=True,
                                                 create_graph=True,
                                                 allow_unused=True)[0].view(-1)

        return dydt

    def update_diffusive_fluxes(self):
        # following https://cantera.org/science/flames.html

        sol = self.sol

        sol.update_transport()

        diffusive_fluxes_star = - (sol.density_mass *
                                   (sol.molecular_weights.T / sol.mean_molecular_weight) *
                                   sol.mix_diff_coeffs * self.grad_to_x(sol.X))

        self.diffusive_fluxes = (diffusive_fluxes_star - self.sol.Y.clone()
                                 * diffusive_fluxes_star.sum(dim=1, keepdim=True))

    def update_dYdx(self):
        # following https://cantera.org/science/flames.html

        self.dYdx = (- self.grad_to_x(self.diffusive_fluxes) +
                     self.sol.molecular_weights.T * self.sol.wdot) / self.m_dot

    def update_dTdx(self):
        # following https://cantera.org/science/flames.html

        sol = self.sol

        dT2dx = self.grad_to_x(sol.T)

        dTdx0 = self.grad_to_x(dT2dx * sol.thermal_conductivity)

        MW = sol.molecular_weights.T

        dTdx1 = (dTdx0 - dT2dx * (self.diffusive_fluxes * sol.cp / MW).sum(dim=1, keepdim=True))

        dTdx2 = dTdx1 - (sol.h * MW * sol.wdot).sum(dim=1, keepdim=True)

        self.dTdx = dTdx2 / self.m_dot / sol.cp_mass.T

    def update_rhs(self):

        self.rhs = torch.cat((self.dTdx, self.dYdx), dim=1)
