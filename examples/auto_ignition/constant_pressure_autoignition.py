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

#TODO: the reactorch class seems to be very slow here, will figure out later

import cantera as ct
import numpy as np
import reactorch as rt
from scipy.integrate import solve_ivp
import torch
from torch.autograd.functional import jacobian as jacobian


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
        
        jac_ = jacobian(self.TYdot_jac, TPY, create_graph=False)
        
        return jac_


mech_yaml = '../../data/gri30.yaml'

sol = rt.Solution(mech_yaml=mech_yaml, device=None)

gas = ct.Solution(mech_yaml)


# Initial condition
P = ct.one_atm
gas.TPX = 1001, P, 'H2:2,O2:1,N2:4'
y0 = np.hstack((gas.T, gas.Y))

# Set up objects representing the ODE and the solver
ode = ReactorOde(gas)

# Integrate the equations using Cantera
t_end = 1e-3
states = ct.SolutionArray(gas, 1, extra={'t': [0.0]})
dt = 1e-5
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


sol.gas.TPX = 1001, P, 'H2:2,O2:1,N2:4'
sol.set_pressure(sol.gas.P)
ode_rt = ReactorOdeRT(sol=sol)

print('finish cantera integration')


# Integrate the equations using ReacTorch
states_rt = ct.SolutionArray(sol.gas, 1, extra={'t': [0.0]})
t = 0
ode_success = True
y = y0


while ode_success and t < t_end:
    odesol = solve_ivp(ode_rt,
                       t_span=(t, t + dt),
                       y0=y,
                       method='BDF',
                       vectorized=True, jac=ode_rt.jac)

    t = odesol.t[-1]
    y = odesol.y[:, -1]
    ode_successful = odesol.success

    sol.gas.TPY = odesol.y[0, -1], P, odesol.y[1:, -1]
    states_rt.append(sol.gas.state, t=t)

# Plot the results
try:
    import matplotlib.pyplot as plt
    L1 = plt.plot(states.t, states.T, ls='--', color='r', label='T', lw=1)
    L1_rt = plt.plot(states_rt.t, states_rt.T, ls='-',
                     color='r', label='T_rt', lw=2)
    plt.xlabel('time (s)')
    plt.ylabel('Temperature (K)')

    plt.twinx()
    L2 = plt.plot(states.t, states('OH').Y, ls='--', label='OH', lw=1)
    L2_rt = plt.plot(states_rt.t, states_rt(
        'OH').Y, ls='-', label='OH_rt', lw=1)
    plt.ylabel('Mass Fraction')

    plt.legend(L1+L2+L1_rt+L2_rt, [line.get_label()
                                   for line in L1+L2+L1_rt+L2_rt], loc='lower right')
    plt.show()
except ImportError:
    print('Matplotlib not found. Unable to plot results.')
