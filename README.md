# PyAdvancedControl

Python Codes for Advanced Control

# lqr_sample

This is a sample code of Linear-Quadratic Regulator

![1](https://github.com/AtsushiSakai/PyAdvancedControl/blob/master/lqr_sample/result.png)

# finite_horizon_optimal_control

This is a finite horizon optimal control sample code

![1](https://github.com/AtsushiSakai/PyAdvancedControl/blob/master/finite_horizon_optimal_control/result.png)


# mpc_sample 

This is a sample code of a simple Model Predictive Control (MPC) Simulation

![1](https://github.com/AtsushiSakai/PyAdvancedControl/blob/master/mpc_sample/result.png)

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


