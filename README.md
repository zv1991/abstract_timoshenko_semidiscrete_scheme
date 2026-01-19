# Abstract Timoshenko Semidiscrete Scheme

## Overview

This repository provides a **numerical implementation** of the methods developed in the article:

> **“On the Numerical Treatment of an Abstract Nonlinear System of Coupled Hyperbolic Equations Associated with the Timoshenko Model”**  
> [*Jemal Rogava*](https://github.com/jemal-rogava) — ORCID: https://orcid.org/0000-0001-9460-4283,  
> [*Zurab Vashakidze*](https://github.com/zv1991) — ORCID: https://orcid.org/0000-0001-8736-6213

The source code implements a combination of a **semi-discrete time-stepping scheme** and a **Legendre–Galerkin spectral method** for an abstract system of coupled hyperbolic equations arising from the Timoshenko model.

---

## Mathematical Background

The Timoshenko beam model describes the dynamics of elastic beams by accounting for both **transverse displacement** and **rotation effects**, leading to a coupled system of hyperbolic partial differential equations.

The referenced article develops:
- An **abstract formulation** of the Timoshenko system
- A **Galerkin-based spatial discretization**

This repository implements the **semi-discrete formulation** proposed in the paper using:
- A **time-stepping scheme** applied to the abstract Timoshenko system
- **Legendre–Galerkin spectral discretization in space**
- Numerically evaluated coefficients and forcing terms

---

## Key Features

- Fully **numerical Legendre–Galerkin spectral method**
- Modular and extensible **solver architecture**
- **Testcase-driven execution model** for validation and benchmarking
- Support for **precomputed spectral coefficients**
- Built-in **diagnostics and convergence testing**
- Clear separation between configuration, solvers, and experiments

---

## Repository Structure

📄 [`main_timoshenko_test_runner.py`](main_timoshenko_test_runner.py)  
&nbsp;&nbsp;&nbsp;&nbsp;Main execution entry point

📁 [`setting/`](setting/)  
&nbsp;&nbsp;&nbsp;&nbsp;Numerical and physical parameters

📁 [`solver/`](solver/)  
&nbsp;&nbsp;&nbsp;&nbsp;Semi-discrete Timoshenko solvers

📁 [`testcase_registry/`](testcase_registry/)  
&nbsp;&nbsp;&nbsp;&nbsp;Definition of numerical testcases

📁 [`utils/`](utils/)  
&nbsp;&nbsp;&nbsp;&nbsp;Numerical utilities and helpers

📁 [`precomp_coeffs/`](precomp_coeffs/)  
&nbsp;&nbsp;&nbsp;&nbsp;Precomputed Galerkin coefficients

📁 [`plots/`](plots/)  
&nbsp;&nbsp;&nbsp;&nbsp;Output plots and diagnostics

📁 [`tests/`](tests/)  
&nbsp;&nbsp;&nbsp;&nbsp;Automated numerical tests

---

## Execution Workflow

1. Numerical parameters are defined in [`setting/`](setting/)
2. A testcase is selected from [`testcase_registry/`](testcase_registry/)
3. The selected testcase is executed via the semi-discrete solver
4. Results, diagnostics, and plots are generated

All workflows are directed by the central test runner.

## Acknowledgements

<div align="justify">
The authors of this article wish to extend their gratitude to Dr. [Giorgi Rukhaia](https://github.com/GR1992) for his fruitful remarks during the development of the programming code for the proposed algorithm.
</div>

---

## Getting Started

### Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib

(Additional dependencies are documented in the source code where required.)

### Running a Simulation

```bash
# Clone the repository
git clone https://github.com/zv1991/abstract_timoshenko_semidiscrete_scheme.git
cd abstract_timoshenko_semidiscrete_scheme

# Adjust numerical parameters if needed
vim setting/

# Run the testcase-driven solver
python main_timoshenko_test_runner.py