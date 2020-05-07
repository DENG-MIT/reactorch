## ReacTorch: A Differentiable Reacting Flow Simulation Package in PyTorch

ReacTorch is a package for simulating chemically reacting flows in PyTorch. The capability of auto-differentiation enables us to efficiently compute the derivatives of the solutions to all of the species concentrations (obtaining Jacobian matrix) as well as model parameters (performing sensitivity analysis) at almost no cost. It also natively supports GPU computation with PyTorch. In addition, the capability of differentiating the entire reacting model is the foundation of adopting many recent hybrid physics-neural network algorithms. This package is aimed at providing an easily accessible platform for implementing those emerging hardware and software infrastructures from the deep learning community in chemically reacting flow simulations.

## Installation

```python
python setup.py install
```

## Usage

```python
import reactorch as rt
```

sample code can be found in `test/Solution_test.py`

## Credit

If you use ReacTorch in a publication, we would appreciate if you cited ReacTorch. This helps to improve the reproducibility of your work, as well as giving credit to the many authors who have contributed their time to developing ReacTorch. The recommended citation for ReacTorch is as follows:

    Weiqi Ji, Sili Deng. ReacTorch: A Differentiable Reacting Flow Simulation Package in PyTorch, https://github.com/DENG-MIT/reactorch, 2020.

ReacTorch was initially developed in [Deng Energy and Nanotechnology Group](https://deng.mit.edu) lead by Prof. Sili Deng at MIT.
