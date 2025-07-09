import numpy as np
from casadi import Opti
from inspect import ismethod
from .UtilFunctions import OptiSol, DirectOCP


class Infos:
    """
    This class contains the BVPDAE's numerical solving informations
    :param success: Boolean indicating if the problems is successfully solved
    :param NLSE_infos: class whose attributes gather informations on the Non-Linear Equations Solver
    :param DAE_infos: class whose attributes gather informations on the mesh refinement procedure
    """
    def __init__(self, done):

        self.success = done["success"]
        self.csInfos = done

    def __str__(self):
        print("##########################")
        print(" CASADI Direct Infos")
        print("   Global Success = ", self.success)
        print("          ")
        print(" CASADI infos")
        print(self.csInfos)
        print("  ")


class DirectSolver:
    """
   :param options: options of the solver listed as follows:

       - display: values must be {0, 1, 2, 3} describing verbosity of the display, 0 being no display and 3 being full. Default is 0
       - print: Boolean value authorizing printing iterations of casadi solveur. Default is False
       - method : Discretization method, euler, rk4 or hermite_simpson. Default is hermite_simpson
   """
    def __init__(self, **options):

        if "display" in options.keys():
            self.display = options["display"]
        else:
            self.display = 0
        if "print" in options.keys():
            self.print = options["print"]
        else:
            self.print = False
        if "rk_method" in options.keys() and options["rk_method"] in ["rk4, euler", "hermite_simpson"]:
            self.rk_method = options["rk_method"]
        else:
            self.rk_method = "hermite_simpson"
        # Initialization of the solutions variables
        self.done = None  # output data concerning the problem solving
        self.time_cas, self.x_cas, self.u_cas, self.prb, self.sol, self.para_cas = None, None, None, None, None, None
        self.time_val, self.x_val, self.u_val, self.para_val, self.OCP = None, None, None, None, None

    def opti(self, optisol: OptiSol, OCP: DirectOCP):
        """
        This methods builds a casadi optimization problem corresponding to the OCP provided in parameter OCP

        :param OCP: Object instanciation of optimal control problem.
        :param opti_parameters: Optional additional optimization parameters
        """
        time, x_val, u_val, opti_parameters = (
            optisol.time.copy(), optisol.ode_sol.copy(), optisol.alg_sol.copy(), optisol.params
        )
        prb = Opti()
        if opti_parameters is not None:
            if not hasattr(opti_parameters, '__iter__'):
                raise Exception("The additional optimization parameters must be provided as an iterable")
            for para in opti_parameters:
                if not isinstance(para, float):
                    raise Exception("Each element of opti_parameters must be a float")
            opti_parameters = opti_parameters.copy()
            para = prb.variable(len(opti_parameters))
            prb.set_initial(para, opti_parameters)
        else:
            para = None
        t = prb.parameter(time.size)
        prb.set_value(t, time)
        x = prb.variable(x_val.shape[0], x_val.shape[1])
        u = prb.variable(u_val.shape[0], u_val.shape[1])
        q = prb.variable(1, x_val.shape[1])

        prb.set_initial(x, x_val)
        prb.set_initial(u, u_val)
        prb.set_initial(q, np.zeros((1, len(time))))

        is_terminal_cost = hasattr(OCP, "terminal_cost") and ismethod(getattr(OCP, "terminal_cost"))
        if is_terminal_cost:
            prb.minimize(q[-1] + OCP.terminal_cost(x[:, -1]))
        else:
            prb.minimize(q[-1])

        is_eq_const = hasattr(OCP, "const_eq") and ismethod(getattr(OCP, "const_eq"))
        is_ineq_const = hasattr(OCP, "const_ineq") and ismethod(getattr(OCP, "const_ineq"))

        for k in range(len(time)):
            if k < len(time) - 1:
                if self.rk_method == "euler":
                    k1 = OCP.ode(t[k], x[:, k], u[:, k])
                    k1_cost = OCP.running_cost(t[k], x[:, k], u[:, k])
                    xkp1 = x[:, k] + (t[k + 1] - t[k]) * k1
                    prb.subject_to(x[:, k + 1] == xkp1)
                    qkp1 = q[:, k] + (t[k + 1] - t[k]) * k1_cost
                    prb.subject_to(q[:, k + 1] == qkp1)
                elif self.rk_method == "rk4":
                    dt = t[k + 1] - t[k]
                    k1 = OCP.ode(
                            t[k], x[:, k], u[:, k]
                        )
                    k2 = OCP.ode(
                        t[k] + .5 * dt, x[:, k] + .5 * dt * k1, .5 * (u[:, k] + u[:, k + 1])
                    )
                    k3 = OCP.ode(
                        t[k] + .5 * dt, x[:, k] + .5 * dt * k2, .5 * (u[:, k] + u[:, k + 1])
                    )
                    k4 = OCP.ode(
                        t[k + 1], x[:, k] + dt * k3, u[:, k + 1]
                    )
                    k1_cost = OCP.running_cost(
                        t[k], x[:, k], u[:, k]
                    )
                    k2_cost = OCP.running_cost(
                        t[k] + .5 * dt, x[:, k] + .5 * dt * k1, .5 * (u[:, k] + u[:, k + 1])
                    )
                    k3_cost = OCP.running_cost(
                        t[k] + .5 * dt, x[:, k] + .5 * dt * k2, .5 * (u[:, k] + u[:, k + 1])
                    )
                    k4_cost = OCP.running_cost(
                        t[k + 1], x[:, k] + dt * k3, u[:, k + 1]
                    )
                    xkp1 = x[:, k] + dt / 6. * (k1 + 2 * k2 + 2 * k3 + k4)
                    prb.subject_to(x[:, k + 1] == xkp1)

                    qkp1 = q[:, k] + dt / 6. * (k1_cost + 2 * k2_cost + 2 * k3_cost + k4_cost)
                    prb.subject_to(q[:, k + 1] == qkp1)
                else:
                    rhsk = OCP.ode(
                        t[k], x[:, k], u[:, k]
                    )
                    rhskp1 = OCP.ode(
                        t[k + 1], x[:, k + 1], u[:, k + 1]
                    )
                    costk = OCP.running_cost(
                        t[k], x[:, k], u[:, k]
                    )
                    costkp1 = OCP.running_cost(
                        t[k+1], x[:, k + 1], u[:, k + 1]
                    )
                    xp05 = .5 * (x[:, k] + x[:, k + 1]) - .125 * (t[k + 1] - t[k]) * (rhskp1 - rhsk)
                    zp05 = .5 * (u[:, k] + u[:, k + 1])
                    tp05 = (t[k + 1] + t[k]) * .5
                    rhskp05 = OCP.ode(tp05, xp05, zp05)
                    costp05 = OCP.running_cost(tp05, xp05, zp05)
                    xkp1 = x[:, k] + (rhskp1 + 4. * rhskp05 + rhsk) * (t[k + 1] - t[k]) / 6.
                    prb.subject_to(x[:, k + 1] == xkp1)
                    qkp1 = q[:, k] + (costkp1 + 4. * costp05 + costk) * (t[k + 1] - t[k]) / 6.
                    prb.subject_to(q[:, k + 1] == qkp1)
            if is_eq_const:
                prb.subject_to(OCP.const_eq(t[k], x[:, k], u[:, k]) == 0.)
            if is_ineq_const:
                prb.subject_to(OCP.const_ineq(t[k], x[:, k], u[:, k]) <= 0.)
        prb.subject_to(q[:, 0] == 0.)
        if para is None:
            bc = OCP.twobc(x[:, 0], x[:, -1], u[:, 0], u[:, -1])
        else:
            bc = OCP.twobc(x[:, 0], x[:, -1], u[:, 0], u[:, -1], para)
        prb.subject_to(bc == 0.)
        options = {"print_time": self.print, "ipopt": {"print_level": self.display}}
        prb.solver("ipopt", options)
        self.prb, self.time_cas, self.x_cas, self.u_cas, self.para_cas = prb, t, x, u, para
        self.time_val, self.x_val, self.u_val, self.para_val, self.OCP = time, x_val, u_val, opti_parameters, OCP

    def update_solutions(self, optisol):
        """
        This methods initialize the optimization variables to the values given as parameters

        :param time: numpy array containing the time steps of the OCP
        :param x: numpy array containing the state solution of the OCP
        :param u: numpy array containing the control solution of the OCP
        :param opti_parameters: iterable of float containing additional optimization parameters
        """
        self.time_val = optisol.time.copy()
        self.x_val = optisol.ode_sol.copy()
        self.u_val = optisol.alg_sol.copy()
        self.prb.set_value(self.time_cas, self.time_val)
        self.prb.set_initial(self.x_cas, self.x_val)
        self.prb.set_initial(self.u_cas, self.u_val)
        if optisol.params is not None:
            self.prb.set_initial(self.para_cas, optisol.params.copy())

    def solve(self,):
        """
        This method solves the problem using IPOPT and returns the solution of OCP as follows

        :return: optimal solution of the problem
                (time, x, u, done) if opti_parameters is None
                (time, x, u, opti_para, done) otherwise
        """
        self.sol = self.prb.solve()
        done = self.sol.stats()
        infos = Infos(done)
        self.x_val = self.sol.value(self.x_cas)
        self.u_val = self.sol.value(self.u_cas)
        if self.para_cas is not None:
            self.para_val = self.sol.value(self.para_cas)
        if len(self.u_val.shape) == 1:
            self.u_val = np.reshape(self.u_val, (1, len(self.u_val)))
        if self.para_cas is not None:
            self.para_val = self.sol.value(self.para_cas)
            optisol = OptiSol(time=self.time_val, ode_sol=self.x_val, alg_sol=self.u_val, params=self.para_val)
        else:
            optisol = OptiSol(time=self.time_val, ode_sol=self.x_val, alg_sol=self.u_val)
        return optisol, infos

    def get_solution(self):
        """
        This method returns the numerical solution of the optimal control problem as follows

        :return: time, x, u, [opti_parameters]
        """
        if self.para_val is None:
            return self.time_val, self.x_val, self.u_val
        else:
            return self.time_val, self.x_val, self.u_val, self.para_val
