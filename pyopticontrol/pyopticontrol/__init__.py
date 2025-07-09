"""
pyopticontrol is a package for solving Optimal Control Problems (OCPs). The solving methodology is automatically derived from
the structure of the optimal control problem given as a parameter when instanciating a EMSOpti object.

- **Instructions for OCP structure**
    - **Direct Optimization using casadi :**
        In this case the input parameter OCP must embed the following methods :
            - **Required methods** :
                - **ode(time, x, u, [opti_parameters])** -> returns the time derivative of vector x
                - **running_cost(time, x, u, [opti_parameters])** -> returns the time derivative of the cost to minimized
                - **twobc(x0, xT, u0, uT, [opti_parameters])** -> returns the boundary conditions of the OCP
            - **Optional methods**
                - **const_ineq(time, x, u, [opti_parameters])** -> returns c(time, x, u, [opti_parameters]) st c <= 0.
                - **const_eq(time, x, u, [opti_parameters])** -> returns c(time, x, u, [opti_parameters]) st c == 0.
                - **terminal_cost(xT)** -> returns the terminal cost part of the objective function as a function of the state at final time

    - **Indirect Optimization using casadi :**
        In this situation the input parameter OCP must embed the following methods :
            - **ode(t, x, z):** -> returns a casadi vector of the size of x containing the value of the derivative of x
            - **algeq(t, x, z):** -> returns a casadi vector of the size of z containing the algebraic equations.
            - **twobc(x0, xT, z0, zT):** -> returns a casadi vector of the boundary conditions.

    - **Indirect Optimization with NumPy Jacobian :**
        In this situation the input parameter OCP must embed the following methods :
            - **ode(t, x, z):** -> returns an array of the size of x containing the value of the derivative of x at each time stamp
            - **algeq(t, x, z):** -> returns an array of the size of z containing the algebraic equations at each time stamp. The solution of the problem corresponds to finding z such that the algebraic equations are 0.
            - **twobc(x0, xT, z0, zT):** -> returns the boundary conditions. The solution of the problem consists in finding the boundary values such that twobc is 0.
            - **odejac(time, x, z):** -> returns (Jx, Jz) two 3D arrays such that :
                    - **Jx:** NumPy array of size (x.shape[0], x.shape[0], len(time)). Jx[:, :, i] contains the Jacobian of the ODEs with respect to x at time t[i]
                    - **Jz:** NumPy array of size (x.shape[0], z.shape[0], len(time)). Jz[:, :, i] contains the Jacobian of the ODEs with respect to z at time t[i]
            - **algjac(time, x, z):** -> returns (Gx, Gz) two 3D arrays such that:
                    - **Gx:** NumPy array of size (z.shape[0], x.shape[0], len(time)). Gx[:, :, i] contains the Jacobian of the Algebraic Equations with respect to x at time t[i]
                    - **Gz:** NumPy array of size (z.shape[0], z.shape[0], len(time)). Gz[:, :, i] contains the Jacobian of the Algebraic Equations with respect to z at time t[i]
            - **bcjac(x0, xT, z0, zT):** -> returns four arrays containing the Jacobians of the boundary conditions as follows
                - **BCx0:** Array of size (x0.shape[0], x0.shape[0]) contains the Jacobian of the boundary conditions with respect to x0
                - **BCxT:** Array of size (xT.shape[0], xT.shape[0]) contains the Jacobian of the boundary conditions with respect to xT
                - **BCz0:** Array of size (x0.shape[0], z0.shape[0]) contains the Jacobian of the boundary conditions with respect to z0
                - **BCzT:** Array of size (x0.shape[0], zT.shape[0]) contains the Jacobian of the boundary conditions with respect to zT
"""

__version__ = "3.1"

from .DAESolver import BVPDAE
from .CSDAESolver import OCPCASADI, OCPCASADIParams
from .UtilFunctions import (repmat, matmul3d, max_smooth, min_smooth, logpen, FB, csFB, OptiSol, IndirectOCP,
                            CSIndirectOCP, DirectOCP)
from .DirectSolver import DirectSolver
from .OptiControl import OptiControl
