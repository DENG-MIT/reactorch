#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Weiqi Ji"
__copyright__ = "Copyright 2020, DENG"

__version__ = "0.1"
__email__ = "weiqiji@mit.edu"
__status__ = "Development"

import torch


torch.set_default_tensor_type("torch.DoubleTensor")


class Flame_1d:
    def __init__(self, tct, device=None):
        super().__init__()
        
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
        
        self.tct = tct

    def set_states(self, TPY):

        self.tct.set_states(TPY)

        self.update_u()
        self.update_diffusive_fluxes()
        self.update_dTdx()
        self.update_dYdx()
        self.update_rhs()

    def set_x(self, x):
        self.x = x

    def set_mdot(self, m_dot):
        self.m_dot = torch.Tensor([m_dot]).to(self.device)

    def update_u(self):
        self.u = self.m_dot / self.tct.density_mass

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
        self.tct.update_transport()
        self.diffusive_fluxes_star = self.tct.density_mass * \
            (self.tct.molecular_weights.T / self.tct.mean_molecular_weight) \
            * self.tct.mix_diff_coeffs * self.grad_to_x(self.tct.X)
        self.diffusive_fluxes = self.diffusive_fluxes_star - \
            self.tct.Y.clone() * self.diffusive_fluxes_star.sum(dim=1, keepdim=True)

    def update_dYdx(self):
        # following https://cantera.org/science/flames.html
        self.dYdx = - (self.grad_to_x(self.diffusive_fluxes) + self.tct.molecular_weights.T * self.tct.wdot) / self.m_dot

    def update_dTdx(self):
        # following https://cantera.org/science/flames.html
        self.dT2dx = self.grad_to_x(self.tct.T)

        self.dTdx0 = self.grad_to_x(self.dT2dx * self.tct.thermal_conductivity)
        self.dTdx1 = self.dTdx0 - self.dT2dx * (self.diffusive_fluxes * self.tct.cp).sum(dim=1, keepdim=True)
        self.dTdx2 = self.dTdx1 - (self.tct.h * self.tct.molecular_weights.T * self.tct.wdot).sum(dim=1, keepdim=True)
        self.dTdx = self.dTdx2/self.m_dot / self.tct.cp_mole.unsqueeze(-1)

    def update_rhs(self):
        self.rhs = torch.cat((self.dTdx, self.dYdx), dim=1)

