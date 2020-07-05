import torch

def forward_rate_constants_func(self):
        """Update forward_rate_constants
        """

        ln10 = torch.log(torch.Tensor([10.0])).to(self.device)
        
        self.forward_rate_constants = self.Arrhenius_A * \
               torch.exp(self.Arrhenius_b * torch.log(self.T) - \
                         self.Arrhenius_Ea * 4184.0 / self.R / self.T) \
                         * self.C_M2
    
        for i in self.list_reaction_type4:
            reaction = self.reaction[i]

            # high pressure
            self.kinf = reaction['A'] * \
                 torch.exp(reaction['b'] * torch.log(self.T) \
                    - reaction['Ea'] * 4184.0 / self.R / self.T)
                
            # low pressure
            self.k0 = self.reaction[i]['A_0'] * \
                 torch.exp(reaction['b_0'] * torch.log(self.T) \
                    - reaction['Ea_0'] * 4184.0 / self.R / self.T)

            Pr = self.k0 * self.C_M[:, i: i + 1] / self.kinf
            lPr = torch.log10(Pr)
            
            self.k = self.kinf * Pr / (1 + Pr)

            if 'Troe' in self.reaction[i]:
                   A = reaction['Troe']['A']
                   T1 = reaction['Troe']['T1']
                   T2 = reaction['Troe']['T2']
                   T3 = reaction['Troe']['T3']

                   F_cent = (1 - A) * torch.exp(-self.T / T3) + \
                        A * torch.exp(-self.T / T1) + torch.exp(-T2 / self.T)

                   lF_cent = torch.log10(F_cent)
                   C = -0.4 - 0.67 * lF_cent
                   N = 0.75 - 1.27 * lF_cent
                   f1 = (lPr + C) / (N - 0.14 * (lPr + C))
                                      
                   F = torch.exp(ln10 * lF_cent / (1 + f1 * f1))
                   self.k = self.k * F
                      
            self.forward_rate_constants[:, i: i + 1] = self.k
        
        for i in self.list_reaction_type5:

            reaction = self.reaction[i]

            self.kk = [[None]] * self.n_rate_constants[i]

            # calculate rate expressions at all given pressures
            for j in range(self.n_rate_constants[i]):
                self.kk[j] = reaction['p_dep']['A'][j] * \
                             torch.exp(reaction['b'][j] * torch.log(self.T) \
                                       - reaction['Ea'][j] * 4184.0 / self.R / self.T)

            # jhigh1 corresponds to the first Arrhenius expression given at the minumum pressure
            # higher than actual pressure. Considering multiple rate expressions may be given
            # at the same pressure, we need jhigh2, which corresponds to the last Arrhenius
            # expression given at the minumum pressure higher than actual pressure.
            jhigh1 = self.n_rate_constants[i]
            for j in range(self.n_rate_constants[i]):
                if self.P[0] <= reaction['P'][j]:
                    jhigh1 = j
                    break

            if jhigh1 != self.n_rate_constants[i]:
                for j in range(self.n_rate_constants[i] - 1, -1, -1):
                    if reaction['P'][j] == reaction['P'][jhigh1]:
                        jhigh2 = j
                        break

            # jlow1 corresponds to the last Arrhenius expression given at the maximum pressure
            # lower than actual pressure while jlow2 corresponds to the first.
            jlow1 = -1
            for j in range(self.n_rate_constants[i] - 1, -1, -1):
                if self.P[0] >= reaction['P'][j]:
                    jlow1 = j
                    break

            if jlow1 != -1:
                for j in range(self.n_rate_constants[i]):
                    if reaction['P'][j] == reaction['P'][jlow1]:
                        jlow2 = j
                        break

            # This is the case where the actual pressure is higher than all given pressures.
            if jhigh1 == self.n_rate_constants[i]:
                for j in range(self.n_rate_constants[i]):
                    if reaction['P'][j] == reaction['P'][jhigh1 - 1]:
                        jhigh2 = j
                        break
                self.k = self.kk[jhigh1 - 1]
                if jhigh2 != jhigh1 - 1:
                    for j in range(jhigh2, jhigh1 - 1):
                        self.k = self.k + self.kk[j]

            # This is the case where the actual pressure is lower than all given pressures.
            if jlow1 == -1:
                for j in range(self.n_rate_constants[i] - 1, -1, -1):
                    if reaction['P'][j] == reaction['P'][0]:
                        jlow2 = j
                        break
                self.k = self.kk[0]
                if jlow2 != 0:
                    for j in range(1, jlow2 + 1):
                        self.k = self.k + self.kk[j]

            # This is the case where the actual pressure is higher than the minimum
            # given pressure and lower than the maximum given pressure.
            if jhigh1 != self.n_rate_constants[i] and jlow1 != -1:
                self.k1 = self.kk[jlow1]
                self.k2 = self.kk[jhigh1]
                if jhigh1 != jhigh2:
                    for j in range(jhigh1 + 1, jhigh2 + 1):
                        self.k2 = self.k2 + self.kk[j]
                if jlow1 != jlow2:
                    for j in range(jlow2, jlow1):
                        self.k1 = self.k1 + self.kk[j]
                logk = torch.log(self.k1) + (torch.log(self.k2) - torch.log(self.k1)) \
                       * (torch.log(self.P[0]) - torch.log(reaction['P'][jlow1])) / \
                       (torch.log(reaction['P'][jhigh1]) - torch.log(reaction['P'][jlow1]))
                self.k = torch.exp(logk)

            self.forward_rate_constants[:, i: i + 1] = self.k

        self.forward_rate_constants = self.forward_rate_constants * self.uq_A.abs()

def forward_rate_constants_func_vec(self):
        """Update forward_rate_constants
        """

        ln10 = torch.log(torch.Tensor([10.0])).to(self.device)
        
        self.forward_rate_constants = self.Arrhenius_A * \
               torch.exp(self.Arrhenius_b * torch.log(self.T) - \
                         self.Arrhenius_Ea * 4184.0 / self.R / self.T) \
                         * self.C_M2
        
        # dealing with type 4
        self.k0 = self.Arrhenius_A0 * \
               torch.exp(self.Arrhenius_b0 * torch.log(self.T) - \
                         self.Arrhenius_Ea0 * 4184.0 / self.R / self.T)   
        self.kinf = self.Arrhenius_Ainf * \
               torch.exp(self.Arrhenius_binf * torch.log(self.T) - \
                         self.Arrhenius_Eainf * 4184.0 / self.R / self.T)  
        Pr = self.k0 * self.C_M_type4 / self.kinf
        
        # transfer the size to match the overall rate matrix
        self.falloff_matrix = torch.mm(Pr / (1 + Pr),self.mat_transfer_type4) + \
            (1 - self.is_falloff) * self.identity_mat
            
        
        # dealing with Troe
        lPr = torch.mm(torch.log10(Pr),self.mat_transfer_type4_to_Troe)
        F_cent = (1 - self.Troe_A) * torch.exp(-self.T / self.Troe_T3) + \
                        self.Troe_A * torch.exp(-self.T / self.Troe_T1) + \
                        torch.exp(-self.Troe_T2 / self.T)
                        
        lF_cent = torch.log10(F_cent)
        C = -0.4 - 0.67 * lF_cent
        N = 0.75 - 1.27 * lF_cent
        f1 = (lPr + C) / (N - 0.14 * (lPr + C))
        
        F = torch.exp(ln10 * lF_cent / (1 + f1 * f1))
        
        # transfer the size to match the overall rate matrix
        self.Troe_matrix = torch.mm(F,self.mat_transfer_type4_Troe) + \
            (1 - self.is_Troe_falloff) * self.identity_mat
            
        self.forward_rate_constants = \
            self.forward_rate_constants * self.Troe_matrix * self.falloff_matrix

        # for uncertainty quantification
        self.forward_rate_constants = self.forward_rate_constants * self.uq_A.abs()

def forward_rate_constants_func_matrix(self):

        self.kf = self.Arrhenius_A * \
            torch.exp(self.Arrhenius_b * torch.log(self.T) - self.Arrhenius_Ea  * 4184.0 / self.R / self.T)

def equilibrium_constants_func(self):

        vk = (-self.reactant_stoich_coeffs + self.product_stoich_coeffs)
        delta_S_over_R = torch.mm(self.S0, vk) / self.R
        delta_H_over_RT = torch.mm(self.H, vk) / self.R / self.T

        self.equilibrium_constants = \
            torch.exp(delta_S_over_R - delta_H_over_RT + torch.log(self.P_atm / self.R / self.T) * vk.sum(dim=0))

def reverse_rate_constants_func(self):

        self.reverse_rate_constants = self.forward_rate_constants / \
            self.equilibrium_constants * self.is_reversible

def wdot_func(self):

        eps = 1e-300

        self.forward_rates_of_progress = self.forward_rate_constants * \
            torch.exp(torch.mm(torch.log(self.C + eps), self.reactant_orders))

        self.reverse_rates_of_progress = self.reverse_rate_constants * \
            torch.exp(torch.mm(torch.log(self.C + eps),
                               self.product_stoich_coeffs))

        self.qdot = self.forward_rates_of_progress - self.reverse_rates_of_progress

        self.wdot = torch.mm(self.qdot, self.net_stoich_coeffs.T)

        self.net_production_rates = self.wdot
