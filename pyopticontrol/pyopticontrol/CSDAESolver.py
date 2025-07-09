import numpy as np
import casadi as cs


__all__ = ["OCPCASADI", "OCPCASADIParams"]


def build_running_fun_and_jacobian(fun, ode_sol_val, alg_sol_val):
    # Initialize symbolic variables
    t = cs.SX.sym('t', 1)
    ode_sol = cs.SX.sym('ode_sol', ode_sol_val.shape[0])
    alg_sol = cs.SX.sym('alg_sol', alg_sol_val.shape[0])

    # create CASADi function
    cfun = fun(t, ode_sol, alg_sol)
    F = cs.Function('F', [t, ode_sol, alg_sol], [cfun])

    # Compute the Jacobian of f with respect to ode_sol
    jac_wrt_ode_sol = cs.jacobian(cfun, ode_sol)
    # Create a CASADi function for the Jacobian
    jac_wrt_ode_sol_func = cs.Function('jac_wrt_ode_sol_func', [t, ode_sol, alg_sol], [jac_wrt_ode_sol])

    # Compute the Jacobian of f with respect to alg_sol
    jac_wrt_alg_sol = cs.jacobian(cfun, alg_sol)
    # Create a CASADi function for the Jacobian
    jac_wrt_alg_sol_func = cs.Function('jac_wrt_alg_sol_func', [t, ode_sol, alg_sol], [jac_wrt_alg_sol])

    return F, jac_wrt_ode_sol_func, jac_wrt_alg_sol_func


def build_bc_fun_and_jacobian(twobc, ode_sol_0_val, ode_sol_T_val, alg_sol_0_val, alg_sol_T_val):
    # Initialize symbolic variables
    ode_sol_0 = cs.SX.sym('ode_sol_0', ode_sol_0_val.size)
    ode_sol_T = cs.SX.sym('ode_sol_T', ode_sol_T_val.size)
    alg_sol_0 = cs.SX.sym('alg_sol_0', alg_sol_0_val.size)
    alg_sol_T = cs.SX.sym('alg_sol_T', alg_sol_T_val.size)

    # create CASADi function
    ctwobc = twobc(ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T)
    twobc_func = cs.Function('twobc_func', [ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T], [ctwobc])

    # Compute the Jacobian of f with respect to ode_sol_0
    jacbc_wrt_ode_sol_0 = cs.jacobian(ctwobc, ode_sol_0)

    # Create a CASADi function for the Jacobian
    jacbc_wrt_ode_sol_0_func = cs.Function(
        'jacbc_wrt_ode_sol_0_func', [ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T], [jacbc_wrt_ode_sol_0]
    )

    # Compute the Jacobian of f with respect to ode_sol_T
    jacbc_wrt_ode_sol_T = cs.jacobian(ctwobc, ode_sol_T)

    # Create a CASADi function for the Jacobian
    jacbc_wrt_ode_sol_T_func = cs.Function(
        'jacbc_wrt_ode_sol_T_func', [ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T], [jacbc_wrt_ode_sol_T]
    )

    # Compute the Jacobian of f with respect to alg_sol_0
    jacbc_wrt_alg_sol_0 = cs.jacobian(ctwobc, alg_sol_0)

    # Create a CASADi function for the Jacobian
    jacbc_wrt_alg_sol_0_func = cs.Function(
        'jacbc_wrt_alg_sol_0_func', [ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T], [jacbc_wrt_alg_sol_0]
    )

    # Compute the Jacobian of f with respect to alg_sol_T
    jacbc_wrt_alg_sol_T = cs.jacobian(ctwobc, alg_sol_T)

    # Create a CASADi function for the Jacobian
    jacbc_wrt_alg_sol_T_func = cs.Function(
        'jacbc_wrt_alg_sol_T_func', [ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T], [jacbc_wrt_alg_sol_T]
    )

    return (
        twobc_func, jacbc_wrt_ode_sol_0_func, jacbc_wrt_ode_sol_T_func, jacbc_wrt_alg_sol_0_func,
        jacbc_wrt_alg_sol_T_func
    )


def build_bc_fun_and_jacobian_params(twobc, ode_sol_0_val, ode_sol_T_val, alg_sol_0_val, alg_sol_T_val, params_val):
    # Initialize symbolic variables
    ode_sol_0 = cs.SX.sym('ode_sol_0', ode_sol_0_val.size)
    ode_sol_T = cs.SX.sym('ode_sol_T', ode_sol_T_val.size)
    alg_sol_0 = cs.SX.sym('alg_sol_0', alg_sol_0_val.size)
    alg_sol_T = cs.SX.sym('alg_sol_T', alg_sol_T_val.size)
    params = cs.SX.sym('params', params_val.size)
    # create CASADi function
    ctwobc = twobc(ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T, params)
    twobc_func = cs.Function('twobc_func', [ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T, params], [ctwobc])

    # Compute the Jacobian of f with respect to ode_sol_0
    jacbc_wrt_ode_sol_0 = cs.jacobian(ctwobc, ode_sol_0)

    # Create a CASADi function for the Jacobian
    jacbc_wrt_ode_sol_0_func = cs.Function(
        'jacbc_wrt_ode_sol_0_func', [ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T, params], [jacbc_wrt_ode_sol_0]
    )

    # Compute the Jacobian of f with respect to ode_sol_T
    jacbc_wrt_ode_sol_T = cs.jacobian(ctwobc, ode_sol_T)

    # Create a CASADi function for the Jacobian
    jacbc_wrt_ode_sol_T_func = cs.Function(
        'jacbc_wrt_ode_sol_T_func', [ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T, params], [jacbc_wrt_ode_sol_T]
    )

    # Compute the Jacobian of f with respect to alg_sol_0
    jacbc_wrt_alg_sol_0 = cs.jacobian(ctwobc, alg_sol_0)

    # Create a CASADi function for the Jacobian
    jacbc_wrt_alg_sol_0_func = cs.Function(
        'jacbc_wrt_alg_sol_0_func', [ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T, params], [jacbc_wrt_alg_sol_0]
    )

    # Compute the Jacobian of f with respect to alg_sol_T
    jacbc_wrt_alg_sol_T = cs.jacobian(ctwobc, alg_sol_T)

    # Create a CASADi function for the Jacobian
    jacbc_wrt_alg_sol_T_func = cs.Function(
        'jacbc_wrt_alg_sol_T_func', [ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T, params], [jacbc_wrt_alg_sol_T]
    )

    # Compute the Jacobian of f with respect to params
    jacbc_wrt_params = cs.jacobian(ctwobc, params)

    # Create a CASADi function for the Jacobian
    jacbc_wrt_params_func = cs.Function(
        'jacbc_wrt_params', [ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T, params], [jacbc_wrt_params]
    )

    return (
        twobc_func, jacbc_wrt_ode_sol_0_func, jacbc_wrt_ode_sol_T_func, jacbc_wrt_alg_sol_0_func,
        jacbc_wrt_alg_sol_T_func, jacbc_wrt_params_func
    )


class OCPCASADI:
    """This class links a TPBVP written in the Casadi framework into the NumPy framework to be solved using the BVPDAE class.

        :param ocp: OCP written in the Casadi framework, returning the right-hand side of the ODEs, the algebraic equations, and the boundary conditions.
        :param ode_sol: NumPy array containing the ODEs' solution.
        :param alg_sol: NumPy array containing the solution of the algebraic equations.
    """
    def __init__(self, ocp, ode_sol, alg_sol):

        self._ode, self._jacode_wrt_ode_sol, self._jacode_wrt_alg_sol = build_running_fun_and_jacobian(
            ocp.ode, ode_sol, alg_sol
        )

        self._algeq, self._jacalg_wrt_ode_sol, self._jacalg_wrt_alg_sol = build_running_fun_and_jacobian(
            ocp.algeq, ode_sol, alg_sol
        )
        self._twobc, self._jacbc_ode_sol_0, self._jacbc_ode_sol_T, self._jacbc_alg_sol_0, self._jacbc_alg_sol_T = (
            build_bc_fun_and_jacobian(ocp.twobc, ode_sol[:, 0], ode_sol[:, -1], alg_sol[:, 0], alg_sol[:, -1])
        )

    def ode(self, time, ode_sol, alg_sol):

        """This function returns the ODEs' right hand side in NumPy array format

            :param time: Numpy array of discretization time steps
            :param ode_sol: Numpy array of ODEs' solution at discretization points
            :param alg_sol: Numpy array of algebraic solutions at discretization points
            :return: **rhs_ode**: Numpy array of time derivative of ode_sol at discretization points
        """
        ode_vec = self._ode.map(len(time))
        return np.array(ode_vec(time, ode_sol, alg_sol)).reshape(ode_sol.shape, order="F")

    def algeq(self, time, ode_sol, alg_sol):
        """This function returns the residual of the algbreaic equations

        :param time: Numpy array of time steps at discretization points
        :param ode_sol: 2d Numpy array of ODEs'solution at discretization points
        :param alg_sol: 2d Numpy array of algebraic variables at discretization points
        :return: **res_alg_eq**: 2d Numpy array of residual of algebraic equations
        """
        algeq_vec = self._algeq.map(len(time))
        return np.array(algeq_vec(time, ode_sol, alg_sol)).reshape(alg_sol.shape, order="F")

    def twobc(self, ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T):

        return np.array(self._twobc(ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T)).flatten()

    def odejac(self, time, ode_sol, alg_sol):
        """
        This function returns the Jacobians of the ODEs' right-hand side as three dimensional Numpy arrays
        :param time: Numpy array of discretization time steps
        :param ode_sol: Numpy array of ODEs' solution at discretization points
        :param alg_sol: Numpy array of algebraic solutions at discretization points
        :return:
            - **jac_ode_wrt_ode_sol**: 3D Numpy array of ODE's right hand side Jacobian with respect to ode_sol at discretization points
            - **jac_ode_wrt_ode_sol**: 3D Numpy array of ODE's right hand side Jacobian with respect to alg_sol at discretization points
        """
        jacode_wrt_ode_sol_vectorized = self._jacode_wrt_ode_sol.map(len(time))
        jacode_wrt_alg_sol_vectorized = self._jacode_wrt_alg_sol.map(len(time))
        return (
            np.array(jacode_wrt_ode_sol_vectorized(time, ode_sol, alg_sol)).reshape(
                ode_sol.shape[0], ode_sol.shape[0], ode_sol.shape[1], order="F"
            ),
            np.array(jacode_wrt_alg_sol_vectorized(time, ode_sol, alg_sol)).reshape(
                ode_sol.shape[0], alg_sol.shape[0], ode_sol.shape[1], order="F"
            )
        )

    def algjac(self, time, ode_sol, alg_sol):
        """This function returns the Jacobians of the algebraix equations as three dimensional Numpy arrays

        :param time: Numpy array of discretization time steps
        :param ode_sol: Numpy array of ODEs' solution at discretization points
        :param alg_sol: Numpy array of algebraic solutions at discretization points
        :return:
            - **jac_alg_wrt_ode_sol**: 3D Numpy array of algebraic equations Jacobian with respect to ode_sol at discretization points
            - **jac_alg_wrt_alg_sol**: 3D Numpy array of algebraic equations Jacobian with respect to alg_sol at discretization points
        """
        jacalg_wrt_ode_sol_vectorized = self._jacalg_wrt_ode_sol.map(len(time))
        jacalg_wrt_alg_sol_vectorized = self._jacalg_wrt_alg_sol.map(len(time))
        return (
            np.array(jacalg_wrt_ode_sol_vectorized(time, ode_sol, alg_sol)).reshape(
                alg_sol.shape[0], ode_sol.shape[0], ode_sol.shape[1], order="F"
            ),
            np.array(jacalg_wrt_alg_sol_vectorized(time, ode_sol, alg_sol)).reshape(
                alg_sol.shape[0], alg_sol.shape[0], ode_sol.shape[1], order="F"
            )
        )

    def bcjac(self, ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T):

        return (
            np.array(self._jacbc_ode_sol_0(ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T)),
            np.array(self._jacbc_ode_sol_T(ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T)),
            np.array(self._jacbc_alg_sol_0(ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T)),
            np.array(self._jacbc_alg_sol_T(ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T))
        )


class OCPCASADIParams:
    """This class links a TPBVP written in the Casadi framework into the NumPy framework to be solved using the BVPDAE class.

        :param ocp: OCP written in the Casadi framework, returning the right-hand side of the ODEs, the algebraic equations, and the boundary conditions.
        :param ode_sol: NumPy array containing the ODEs' solution.
        :param alg_sol: NumPy array containing the solution of the algebraic equations.
    """
    def __init__(self, ocp, ode_sol, alg_sol, params):

        self._ode, self._jacode_wrt_ode_sol, self._jacode_wrt_alg_sol = build_running_fun_and_jacobian(
            ocp.ode, ode_sol, alg_sol
        )

        self._algeq, self._jacalg_wrt_ode_sol, self._jacalg_wrt_alg_sol = build_running_fun_and_jacobian(
            ocp.algeq, ode_sol, alg_sol
        )
        (self._twobc, self._jacbc_ode_sol_0, self._jacbc_ode_sol_T, self._jacbc_alg_sol_0, self._jacbc_alg_sol_T,
         self._jacbc_params) = build_bc_fun_and_jacobian_params(
            ocp.twobc, ode_sol[:, 0], ode_sol[:, -1], alg_sol[:, 0], alg_sol[:, -1], params
        )

    def ode(self, time, ode_sol, alg_sol):

        """This function returns the ODEs' right hand side in NumPy array format

            :param time: Numpy array of discretization time steps
            :param ode_sol: Numpy array of ODEs' solution at discretization points
            :param alg_sol: Numpy array of algebraic solutions at discretization points
            :return: **rhs_ode**: Numpy array of time derivative of ode_sol at discretization points
        """
        ode_vec = self._ode.map(len(time))
        return np.array(ode_vec(time, ode_sol, alg_sol)).reshape(ode_sol.shape, order="F")

    def algeq(self, time, ode_sol, alg_sol):
        """This function returns the residual of the algbreaic equations

        :param time: Numpy array of time steps at discretization points
        :param ode_sol: 2d Numpy array of ODEs'solution at discretization points
        :param alg_sol: 2d Numpy array of algebraic variables at discretization points
        :return: **res_alg_eq**: 2d Numpy array of residual of algebraic equations
        """
        algeq_vec = self._algeq.map(len(time))
        return np.array(algeq_vec(time, ode_sol, alg_sol)).reshape(alg_sol.shape, order="F")

    def twobc(self, ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T, params):

        return np.array(self._twobc(ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T, params)).flatten()

    def odejac(self, time, ode_sol, alg_sol):
        """
        This function returns the Jacobians of the ODEs' right-hand side as three dimensional Numpy arrays
        :param time: Numpy array of discretization time steps
        :param ode_sol: Numpy array of ODEs' solution at discretization points
        :param alg_sol: Numpy array of algebraic solutions at discretization points
        :return:
            - **jac_ode_wrt_ode_sol**: 3D Numpy array of ODE's right hand side Jacobian with respect to ode_sol at discretization points
            - **jac_ode_wrt_ode_sol**: 3D Numpy array of ODE's right hand side Jacobian with respect to alg_sol at discretization points
        """
        jacode_wrt_ode_sol_vectorized = self._jacode_wrt_ode_sol.map(len(time))
        jacode_wrt_alg_sol_vectorized = self._jacode_wrt_alg_sol.map(len(time))
        return (
            np.array(jacode_wrt_ode_sol_vectorized(time, ode_sol, alg_sol)).reshape(
                ode_sol.shape[0], ode_sol.shape[0], ode_sol.shape[1], order="F"
            ),
            np.array(jacode_wrt_alg_sol_vectorized(time, ode_sol, alg_sol)).reshape(
                ode_sol.shape[0], alg_sol.shape[0], ode_sol.shape[1], order="F"
            )
        )

    def algjac(self, time, ode_sol, alg_sol):
        """This function returns the Jacobians of the algebraix equations as three dimensional Numpy arrays

        :param time: Numpy array of discretization time steps
        :param ode_sol: Numpy array of ODEs' solution at discretization points
        :param alg_sol: Numpy array of algebraic solutions at discretization points
        :return:
            - **jac_alg_wrt_ode_sol**: 3D Numpy array of algebraic equations Jacobian with respect to ode_sol at discretization points
            - **jac_alg_wrt_alg_sol**: 3D Numpy array of algebraic equations Jacobian with respect to alg_sol at discretization points
        """
        jacalg_wrt_ode_sol_vectorized = self._jacalg_wrt_ode_sol.map(len(time))
        jacalg_wrt_alg_sol_vectorized = self._jacalg_wrt_alg_sol.map(len(time))
        return (
            np.array(jacalg_wrt_ode_sol_vectorized(time, ode_sol, alg_sol)).reshape(
                alg_sol.shape[0], ode_sol.shape[0], ode_sol.shape[1], order="F"
            ),
            np.array(jacalg_wrt_alg_sol_vectorized(time, ode_sol, alg_sol)).reshape(
                alg_sol.shape[0], alg_sol.shape[0], ode_sol.shape[1], order="F"
            )
        )

    def bcjac(self, ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T, params):

        return (
            np.array(self._jacbc_ode_sol_0(ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T, params)),
            np.array(self._jacbc_ode_sol_T(ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T, params)),
            np.array(self._jacbc_alg_sol_0(ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T, params)),
            np.array(self._jacbc_alg_sol_T(ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T, params)),
            np.array(self._jacbc_params(ode_sol_0, ode_sol_T, alg_sol_0, alg_sol_T, params))
        )
