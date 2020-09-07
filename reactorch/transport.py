import torch


def update_transport(self):

    self.viscosities_func()

    self.thermal_conductivity_func()

    self.binary_diff_coeffs_func()


def viscosities_func(self):
    #TODO: investigate why we have to clone X in order to avoid inplace operation
    self.trans_T = torch.cat([self.logT ** i for i in reversed(range(self.poly_order))], dim=1)

    self.species_viscosities = torch.mm(self.trans_T, self.species_viscosities_poly)

    self.Wk_over_Wj = torch.mm(self.molecular_weights, 1 / self.molecular_weights.T)

    self.Wj_over_Wk = 1 / self.Wk_over_Wj

    self.etak_over_etaj = torch.bmm(
        self.species_viscosities.unsqueeze(-1),
        1 / self.species_viscosities.unsqueeze(-1).view(-1, 1, self.n_species))

    self.PHI = (1 / 2.8284271247461903 / torch.sqrt(1 + self.Wk_over_Wj) *
                (1 + self.etak_over_etaj ** 0.5 * self.Wj_over_Wk ** 0.25) ** 2)

    X = self.X.clone()
    self.viscosities = (X * self.species_viscosities /
                        torch.bmm(self.PHI, X.unsqueeze(-1)).squeeze(-1)).sum(dim=1,
                                                                                   keepdim=True)


def thermal_conductivity_func(self):
    #TODO: investigate why we have to clone X in order to avoid inplace operation
    self.species_thermal_conductivity = torch.mm(self.trans_T, self.thermal_conductivity_poly)
    X = self.X.clone()
    self.thermal_conductivity = 0.5 * (
        (X * self.species_thermal_conductivity).sum(dim=1, keepdim=True) +
        1 / (X / self.species_thermal_conductivity).sum(dim=1, keepdim=True)
    )


def binary_diff_coeffs_func(self):

    self.binary_diff_coeffs = torch.mm(
        self.trans_T, self.binary_diff_coeffs_poly).view(-1, self.n_species, self.n_species)

    self.X_eps = self.X.clamp_(1e-12)

    self.XjWj = (torch.mm(self.X_eps, self.molecular_weights) -
                 self.X_eps * self.molecular_weights.T)

    self.XjDjk = torch.bmm(self.X_eps.view(-1, 1, self.n_species), 1 / self.binary_diff_coeffs).squeeze(
        1) - self.X_eps / self.binary_diff_coeffs.diagonal(dim1=-2, dim2=-1)

    self.mix_diff_coeffs = self.XjWj / self.XjDjk / self.mean_molecular_weight / self.P * self.one_atm
