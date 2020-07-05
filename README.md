## ReacTorch: A Differentiable Reacting Flow Simulation Package in PyTorch
[![Documentation Status](https://readthedocs.org/projects/reactorch/badge/?version=latest)](https://reactorch.readthedocs.io/en/latest/?badge=latest)

ReacTorch is a package for simulating chemically reacting flows in PyTorch. The capability of auto-differentiation enables us to efficiently compute the derivatives of the solutions to all of the species concentrations (obtaining Jacobian matrix) as well as model parameters (performing sensitivity analysis) at almost no cost. It also natively supports GPU computation with PyTorch. In addition, the capability of differentiating the entire reacting model is the foundation of adopting many recent hybrid physics-neural network algorithms. This package is aimed at providing an easily accessible platform for implementing those emerging hardware and software infrastructures from the deep learning community in chemically reacting flow simulations.

In case you are wondering what is the [relationshop between ReacTorch and Cantera/Chemkin](https://github.com/DENG-MIT/reactorch/issues/5).

## Installation

```shell
git clone git@github.com:DENG-MIT/reactorch.git
cd reactorch
python setup.py install
```

## Requirements

* PyTorch
* Cantera >= 2.5.0
* ruamel.yaml

Detailed instructions on installing the dependent packages can be found in the [wiki page](https://github.com/DENG-MIT/reactorch/wiki/Installation).

## Usage

```python
import reactorch as rt
```

Sample code can be found in `test/Solution_test.py` and examples folder. For example, the autoignition case demonstrates that you can compute jacobian matrix with only couple lines of code!

## Credit

If you use ReacTorch in a publication, we would appreciate if you cited ReacTorch. This helps to improve the reproducibility of your work, as well as giving credit to the many authors who have contributed their time to developing ReacTorch. The recommended citation for ReacTorch is as follows:

    Weiqi Ji, Sili Deng. ReacTorch: A Differentiable Reacting Flow Simulation Package in PyTorch, https://github.com/DENG-MIT/reactorch, 2020.

ReacTorch was initially developed in [Deng Energy and Nanotechnology Group](https://deng.mit.edu) lead by Prof. Sili Deng at MIT.
