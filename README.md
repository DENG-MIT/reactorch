## ReacTorch: A Differentiable Reacting Flow Simulation Package in PyTorch


## Introduction

The capability of Auto-differentiation enable us to efficiently compute the derivative of the solutions to all of the species concentrations (Jacobian) as well model parameters (Sensitivity analysis) at almost no cost. It also natively support of GPU with PyTorch. In addition, the compatibility of differentiate the entire combustion model is the foundations of many latest hybrid physics-neural network algorithms. This package provides a easy-accessible platform for implementing those emerging hardware and software infrastructure from deep learning community.

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

    Weiqi Ji, Sili Deng. ReacTorch: A Differentiable Reacting Flow Simulation Package in PyTorch,   https://github.com/DENG-MIT/reactorch, 2020.

ReacTorch is initially developed in [Deng Energy and Nanotechnology Group](https://deng.mit.edu) lead by Prof. Sili Deng at MIT.