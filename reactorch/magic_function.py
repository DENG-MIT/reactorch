

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
