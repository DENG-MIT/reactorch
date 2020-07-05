import torch

def set_nasa(self):

        self.nasa_low = torch.zeros([self.n_species, 7]).to(self.device)

        self.nasa_high = torch.zeros([self.n_species, 7]).to(self.device)

        for i in range(self.n_species):

            self.nasa_low[i, :] = torch.Tensor(self.model_yaml['species'][i]['thermo']['data'][0])

            self.nasa_high[i, :] = torch.Tensor(self.model_yaml['species'][i]['thermo']['data'][1])

def set_reactions(self):

        self.reaction = [[None]] * self.n_reactions

        self.reactant_stoich_coeffs = torch.Tensor(self.gas.reactant_stoich_coeffs()).to(self.device)

        self.reactant_orders = torch.Tensor(self.gas.reactant_stoich_coeffs()).to(self.device)

        self.product_stoich_coeffs = torch.Tensor(self.gas.product_stoich_coeffs()).to(self.device)

        self.net_stoich_coeffs = self.product_stoich_coeffs - self.reactant_stoich_coeffs

        self.efficiencies_coeffs = torch.ones([self.n_species, self.n_reactions]).to(self.device)

        self.Arrhenius_coeffs = torch.zeros([self.n_reactions, 3]).to(self.device)

        self.is_reversible = torch.ones([self.n_reactions]).to(self.device)
        
        # reaction type 2 will be set as 1
        self.is_three_body = torch.zeros([self.n_reactions]).to(self.device)
        
        self.is_falloff = torch.zeros([self.n_reactions]).to(self.device)
        
        self.is_Troe_falloff = torch.zeros([self.n_reactions]).to(self.device)

        self.list_reaction_type1 = []
        self.list_reaction_type2 = []
        self.list_reaction_type4 = []
        self.list_reaction_type4_Troe = []
        # the id of Troe in type 4
        self.list_id_Troe_in_type4 = []
        
        # count the num in type 4
        count_type4 = -1

        for i in range(self.n_reactions):

            # Type 1: regular reaction, 2: three-body, 4:fall-off

            yaml_reaction = self.model_yaml['reactions'][i]

            self.reaction[i] = {'equation': self.gas.reaction_equation(i)}
            self.reaction[i]['reactants'] = self.gas.reactants(i)
            self.reaction[i]['products'] = self.gas.products(i)
            self.reaction[i]['reaction_type'] = self.gas.reaction_type(i)

            if self.gas.reaction_type(i) in [1]:
                self.list_reaction_type1.append(i)

            if self.gas.reaction_type(i) in [2]:
                self.list_reaction_type2.append(i)

            if self.gas.reaction_type(i) in [4]:
                # id of the current item in the list
                count_type4 = count_type4 + 1
                self.list_reaction_type4.append(i)

            if self.gas.is_reversible(i) is False:
                self.is_reversible[i].fill_(0)

            if self.gas.reaction_type(i) in [1, 2]:

                self.reaction[i]['A'] = torch.Tensor([yaml_reaction['rate-constant']['A']]).to(self.device)

                self.reaction[i]['b'] = torch.Tensor([yaml_reaction['rate-constant']['b']]).to(self.device)

                if type(yaml_reaction['rate-constant']['Ea']) is str:
                    Ea = list(map(eval, [yaml_reaction['rate-constant']['Ea'].split(' ')[0]]))
                else:
                    Ea = [yaml_reaction['rate-constant']['Ea']]

                self.reaction[i]['Ea'] = torch.Tensor(Ea).to(self.device)

            if self.gas.reaction_type(i) in [2]:

                self.is_three_body[i] = 1
                
                if 'efficiencies' in yaml_reaction:
                
                    self.reaction[i]['efficiencies'] = yaml_reaction['efficiencies']
                
                    for key, value in self.reaction[i]['efficiencies'].items():
                
                        self.efficiencies_coeffs[self.gas.species_index(key), i] = value

            if self.gas.reaction_type(i) in [4]:
 
                self.is_falloff[i] = 1
                   
                if 'efficiencies' in yaml_reaction:
                
                    self.reaction[i]['efficiencies'] = yaml_reaction['efficiencies']
                
                    for key, value in self.reaction[i]['efficiencies'].items():
                
                        self.efficiencies_coeffs[self.gas.species_index(key), i] = value

                high_p = yaml_reaction['high-P-rate-constant']

                low_p = yaml_reaction['low-P-rate-constant']

                self.reaction[i]['A'] = torch.Tensor([high_p['A']]).to(self.device)

                self.reaction[i]['b'] = torch.Tensor([high_p['b']]).to(self.device)

                if type(high_p['Ea']) is str:

                    Ea = list(map(eval, [high_p['Ea'].split(' ')[0]]))

                else:

                    Ea = [high_p['Ea']]

                self.reaction[i]['Ea'] = torch.Tensor(Ea).to(self.device)

                self.reaction[i]['A_0'] = torch.Tensor([low_p['A']]).to(self.device)

                self.reaction[i]['b_0'] = torch.Tensor([low_p['b']]).to(self.device)

                if type(low_p['Ea']) is str:

                    Ea = list(map(eval, [low_p['Ea'].split(' ')[0]]))

                else:

                    Ea = [low_p['Ea']]

                self.reaction[i]['Ea_0'] = torch.Tensor(Ea).to(self.device) 

                if 'Troe' in yaml_reaction:
                    
                    self.is_Troe_falloff[i] = 1
                    
                    self.list_reaction_type4_Troe.append(i)
                    self.list_id_Troe_in_type4.append(count_type4)

                    Troe = yaml_reaction['Troe']

                    if 'T2' in Troe:

                        self.reaction[i]['Troe'] = {'A': torch.Tensor([Troe['A']]).to(self.device),
                                                    'T1': torch.Tensor([Troe['T1']]).to(self.device),
                                                    'T2': torch.Tensor([Troe['T2']]).to(self.device),
                                                    'T3': torch.Tensor([Troe['T3']]).to(self.device)
                                                    }

                    else:

                        self.reaction[i]['Troe'] = {'A': torch.Tensor([Troe['A']]).to(self.device),
                                                    'T1': torch.Tensor([Troe['T1']]).to(self.device),
                                                    'T2': torch.Tensor([1e30]).to(self.device),
                                                    'T3': torch.Tensor([Troe['T3']]).to(self.device)
                                                    }

            if 'orders' in yaml_reaction:

                for key, value in yaml_reaction['orders'].items():

                    self.reactant_orders[self.gas.species_index(key), i] = value

            if 'units' in self.model_yaml:

                if self.model_yaml['units']['length'] == 'cm' and self.model_yaml['units']['quantity'] == 'mol':
                    self.reaction[i]['A'] *= (1e-3) ** (self.reactant_stoich_coeffs[:, i].sum().item() - 1)

                    if self.gas.reaction_type(i) in [2]:
                        self.reaction[i]['A'] *= 1e-3

                    if self.gas.reaction_type(i) in [4]:
                        self.reaction[i]['A_0'] *= 1e-3
                        self.reaction[i]['A_0'] *= (1e-3) ** (self.reactant_stoich_coeffs[:, i].sum().item() - 1) 

            self.Arrhenius_coeffs[i, 0] = self.reaction[i]['A']
            self.Arrhenius_coeffs[i, 1] = self.reaction[i]['b']
            self.Arrhenius_coeffs[i, 2] = self.reaction[i]['Ea']

        self.Arrhenius_A = self.Arrhenius_coeffs[:, 0]
        self.Arrhenius_b = self.Arrhenius_coeffs[:, 1]
        self.Arrhenius_Ea = self.Arrhenius_coeffs[:, 2]
        
        
        if self.vectorize==True:
            # for falloff and Troe
            self.length_type4 = len(self.list_reaction_type4)
            self.Arrhenius_A0 = torch.zeros([self.length_type4]).to(self.device)
            self.Arrhenius_b0 = torch.zeros([self.length_type4]).to(self.device)
            self.Arrhenius_Ea0 = torch.zeros([self.length_type4]).to(self.device)
            
            self.Arrhenius_Ainf = torch.zeros([self.length_type4]).to(self.device)
            self.Arrhenius_binf = torch.zeros([self.length_type4]).to(self.device)
            self.Arrhenius_Eainf = torch.zeros([self.length_type4]).to(self.device)
            
            self.efficiencies_coeffs_type4 = torch.ones([self.n_species,self.length_type4]).to(self.device)
            
            self.length_type4_Troe = len(self.list_reaction_type4_Troe)
            self.Troe_A = torch.zeros([self.length_type4_Troe]).to(self.device)
            self.Troe_T1 = torch.zeros([self.length_type4_Troe]).to(self.device)
            self.Troe_T2 = torch.zeros([self.length_type4_Troe]).to(self.device)
            self.Troe_T3 = torch.zeros([self.length_type4_Troe]).to(self.device)  
            
            # for matrix size transfer
            self.mat_transfer_type4 = torch.zeros([self.length_type4,self.n_reactions])
            self.mat_transfer_type4_Troe = torch.zeros([self.length_type4_Troe,self.n_reactions])
            self.mat_transfer_type4_to_Troe = torch.zeros(self.length_type4,self.length_type4_Troe)
    
        
            for i in range(self.length_type4):
                index = self.list_reaction_type4[i]
                
                self.Arrhenius_A0[i] = self.reaction[index]['A_0']
                self.Arrhenius_b0[i] = self.reaction[index]['b_0']
                self.Arrhenius_Ea0[i] = self.reaction[index]['Ea_0']
                
                self.Arrhenius_Ainf[i] = self.reaction[index]['A']
                self.Arrhenius_binf[i] = self.reaction[index]['b']
                self.Arrhenius_Eainf[i] = self.reaction[index]['Ea']
                
                self.efficiencies_coeffs_type4[:,i] = self.efficiencies_coeffs[:,index]
                
                self.mat_transfer_type4[i,index] = 1
            
            for i in range(self.length_type4_Troe):
                index = self.list_reaction_type4_Troe[i]
                
                self.Troe_A[i] = self.reaction[index]['Troe']['A']
                self.Troe_T1[i] = self.reaction[index]['Troe']['T1']
                self.Troe_T2[i] = self.reaction[index]['Troe']['T2']
                self.Troe_T3[i] = self.reaction[index]['Troe']['T3']
                
                self.mat_transfer_type4_Troe[i,index] = 1
                
                index_in_type4 = self.list_id_Troe_in_type4[i]
                self.mat_transfer_type4_to_Troe[index_in_type4,i] = 1
