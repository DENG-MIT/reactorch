import torch


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

    self.S = self.S0 - self.R * torch.log(self.X) - self.R * torch.log(self.P / self.one_atm)

    self.partial_molar_entropies = self.S

    self.entropy_mole = (self.S * self.X).sum(dim=1)


def entropy_mass_func(self):

    self.s = self.S / self.molecular_weights.T

    self.entropy_mass = (self.s * self.Y).sum(dim=1)
