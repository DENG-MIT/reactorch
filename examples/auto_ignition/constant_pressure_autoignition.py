"""
Solve a constant pressure ignition problem where the governing equations are
implemented in Python.

This demonstrates an approach for solving problems where Cantera's reactor
network model cannot be configured to describe the system in question. Here,
Cantera is used for evaluating thermodynamic properties and kinetic rates while
an external ODE solver is used to integrate the resulting equations. In this
case, the SciPy wrapper for VODE is used, which uses the same variable-order BDF
methods as the Sundials CVODES solver used by Cantera.
"""

# TODO: the reactorch class seems to be very slow here, will figure out later

import cantera as ct
import numpy as np
import reactorch as rt
from scipy.integrate import solve_ivp
import torch
from torch.autograd.functional import jacobian as jacobian
import matplotlib.pyplot as plt


class ReactorOde(object):
    def __init__(self, gas):
        # Parameters of the ODE system and auxiliary data are stored in the
        # ReactorOde object.
        self.gas = gas
        self.P = gas.P

    def __call__(self, t, y):
        """the ODE function, y' = f(t,y) """

        # State vector is [T, Y_1, Y_2, ... Y_K]
        self.gas.set_unnormalized_mass_fractions(y[1:])
        self.gas.TP = y[0], self.P
        rho = self.gas.density

        wdot = self.gas.net_production_rates
        dTdt = - (np.dot(self.gas.partial_molar_enthalpies, wdot) /
                  (rho * self.gas.cp))
        dYdt = wdot * self.gas.molecular_weights / rho

        return np.hstack((dTdt, dYdt))


class ReactorOdeRT(object):
    def __init__(self, sol):
        # Parameters of the ODE system and auxiliary data are stored in the
        # ReactorOde object.
        self.sol = sol
        self.gas = sol.gas

    def __call__(self, t, y):
        """the ODE function, y' = f(t,y) """

        TPY = torch.Tensor(y).T

        with torch.no_grad():

            self.sol.set_states(TPY)

            TYdot = self.sol.TYdot_func()

        return TYdot.T.detach().cpu().numpy()

    def TYdot_jac(self, TPY):

        TPY = torch.Tensor(TPY).unsqueeze(0)

        sol.set_states(TPY)

        return sol.TYdot_func().squeeze(0)

    def jac(self, t, y):

        TPY = torch.Tensor(y)

        TPY.requires_grad = True

        jac_ = jacobian(self.TYdot_jac, TPY, create_graph=False)

        return jac_

mech_yaml = '../../data/gri30.yaml'
gas = ct.Solution(mech_yaml)

# Initial condition
P = ct.one_atm * 20
T = 1300
composition = 'ch4:0.5,O2:1,N2:4'
gas.TPX = T, P, composition
y0 = np.hstack((gas.T, gas.Y))

# Set up objects representing the ODE and the solver
ode = ReactorOde(gas)

# Integrate the equations using Cantera
t_end = 1e-3
states = ct.SolutionArray(gas, 1, extra={'t': [0.0]})
dt = t_end / 50
t = 0
ode_success = True
y = y0
while ode_success and t < t_end:
    odesol = solve_ivp(ode,
                       t_span=(t, t + dt),
                       y0=y,
                       method='BDF',
                       vectorized=False, jac=None)

    t = odesol.t[-1]
    y = odesol.y[:, -1]
    ode_successful = odesol.success

    gas.TPY = odesol.y[0, -1], P, odesol.y[1:, -1]
    states.append(gas.state, t=t)

# inpsect if ignition happened
plt.figure()
plt.plot(states.t, states.T, ls='--', color='r', label='T Cantera', lw=1)
plt.show()

print('finish cantera integration')


sol = rt.Solution(mech_yaml=mech_yaml, device=None, vectorize=True,
                  is_clip=False, is_norm=False, is_wdot_vec=False)

sol.gas.TPX = T, P, composition
sol.set_pressure(sol.gas.P)
ode_rt = ReactorOdeRT(sol=sol)

# Integrate the equations using ReacTorch
states_rt = ct.SolutionArray(sol.gas, 1, extra={'t': [0.0]})
t = 0
ode_success = True
y = y0

# Diable AD for jacobian seems more effient for this case.
while ode_success and t < t_end:
    odesol = solve_ivp(ode_rt,
                       t_span=(t, t + dt),
                       y0=y,
                       method='BDF',
                       vectorized=True, jac=None)

    t = odesol.t[-1]
    y = odesol.y[:, -1]
    ode_successful = odesol.success
    sol.gas.TPY = odesol.y[0, -1], P, odesol.y[1:, -1]
    states_rt.append(sol.gas.state, t=t)

    print('t {:.2e} T {:.6f}'.format(t, y[0]))

# Plot the results
fig = plt.figure(figsize=(9, 4))
plt.subplot(121)
L1 = plt.plot(states.t, states.T, ls='--', label='T Cantera', lw=1)
L1_rt = plt.plot(states_rt.t, states_rt.T, ls='-', label='T ReacTorch', lw=1)
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.legend(loc='best')

plt.subplot(122)
L2 = plt.plot(states.t, states('OH').Y, label='OH Cantera', lw=1)
L2_rt = plt.plot(states_rt.t, states_rt('OH').Y, label='OH ReacTorch', lw=1)
plt.ylabel('Mass Fraction')
plt.xlabel('Time (s)')
plt.legend(loc='best')

fig.tight_layout()
plt.savefig('cantera_reactorch_validation.png', dpi=120)
plt.show()
