# PyAdvancedControl

[![Build Status](https://travis-ci.org/AtsushiSakai/PyAdvancedControl.svg?branch=master)](https://travis-ci.org/AtsushiSakai/PyAdvancedControl)

Python Codes for Advanced Control

# Dependencies

- Python 3.7.x

- cvxpy 1.0.x

- ecos 2.0.7

- cvxopt 1.2.x

- scipy 1.1.0

- numpy 1.15.0

- matplotlib 2.2.2

# lqr_sample

This is a sample code of Linear-Quadratic Regulator

This is LQR regulator simulation.

![1](https://github.com/AtsushiSakai/PyAdvancedControl/blob/master/lqr_sample/Figure_1.png)

This is LQR tracking simulation.

![1](https://github.com/AtsushiSakai/PyAdvancedControl/blob/master/lqr_sample/Figure_2.png)

# finite_horizon_optimal_control

This is a finite horizon optimal control sample code

![1](https://github.com/AtsushiSakai/PyAdvancedControl/blob/master/finite_horizon_optimal_control/result.png)


# mpc_sample 

This is a sample code of a simple Model Predictive Control (MPC) regulator simulation

![1](https://github.com/AtsushiSakai/PyAdvancedControl/blob/master/mpc_sample/result.png)

# mpc_tracking 

This is a sample code of a Model Predictive Control (MPC) traget tracking simulation

![1](https://github.com/AtsushiSakai/PyAdvancedControl/blob/master/mpc_tracking/result1.png)

# mpc_modeling 

This is a sample code for model predictive control optimization modeling without any modeling tool (e.g cvxpy)

This means it only use a solver (cvxopt) for MPC optimization.

It includes two MPC optimization functions:

1 opt_mpc_with_input_const()

It can be applied input constraints (not state constraints).

2 opt_mpc_with_state_const()

It can be applied state constraints and input constraints.

This figure is a comparison of MPC results with and without modeling tool.

![1](https://github.com/AtsushiSakai/PyAdvancedControl/blob/master/mpc_modeling/result.png)


## inverted_pendulum_mpc_control

![1](https://github.com/AtsushiSakai/PyAdvancedControl/blob/master/inverted_pendulum_mpc_control/animation.gif)

This is a inverted pendulum mpc control simulation.


# tools

## c2d

This is a API compatible function of MATLAB c2d function. 

[Convert model from continuous to discrete time MATLAB c2d](https://jp.mathworks.com/help/control/ref/c2d.html)

