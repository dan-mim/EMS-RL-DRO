from inspect import ismethod
from .DAESolver import BVPDAE
from .DAESolver import Infos as InfosBVP
from .DirectSolver import DirectSolver
from .DirectSolver import Infos as InfosDirect
from .CSDAESolver import OCPCASADI, OCPCASADIParams
from .UtilFunctions import IndirectOCP, CSIndirectOCP, DirectOCP, OptiSol
from typing import Union, Optional, Any


__all__ = ['OptiControl']


def has_method(obj, method):
    return hasattr(obj, method) and ismethod(getattr(obj, method))


class OptiControl:
    """
    :param ocp: Object instanciation describing an optimal control problem
    :param options: dictionnary containing the options parametererization of the solver
    """

    def __init__(self,
                 ocp: Union[IndirectOCP, DirectOCP, CSIndirectOCP],
                 options: Optional[dict[str, Any]] = None):

        if options is None:
            options = dict()
        self.OCP = ocp
        # All OCPs must have an ODE method
        if not has_method(self.OCP, "ode"):
            raise Exception("Given OCP has no ode method")

        # If provided, the options must be a dictionnary
        if options is not None and not isinstance(options, dict):
            raise Exception("EMS Opti options must be provided as a dictionary or as a None variable")

        # Tests to see if provided OCP has all jacobian
        is_jacobian = (
                has_method(self.OCP, "odejac") and has_method(self.OCP, "algjac")
                and has_method(self.OCP, "bcjac")
        )

        # If not all jacobians are provided, OCP must be casadi object
        if not is_jacobian:
            # Test if casadi object has a boundary function
            if not has_method(self.OCP, "twobc"):
                raise Exception("Given OCP has no twobc method")
            # OCP must have either a cost or a algeq method
            if not has_method(self.OCP, "algeq"):
                self.case = 0
            else:
                self.case = 1
        else:
            self.case = 2

        if self.case == 0:
            self.solver_direct = DirectSolver(**options)
        else:
            self.solver_bvp = BVPDAE(**options)

    def solve(self, sol: OptiSol) -> tuple[OptiSol, Union[InfosDirect, InfosBVP]]:
        """
        Solve the optimal control problem OCP provided at instanciation of class EMSOpti. If no input parameters are provided, this method calls the initialize() method of OCP problem.

        :param sol: OptiSol object
        :return: Optimal solutions of the problem:
            - **optisol:** OptiSol object containing the solution of the DAEs
            - **infos:** informations on the DAEs solving
        """
        time, ode_sol, alg_sol, alg_sol_mid, params = sol.time, sol.ode_sol, sol.alg_sol, sol.alg_sol_mid, sol.params
        if time is None or ode_sol is None or alg_sol is None:
            raise Exception("time, ode_sol and alg_sol must not be None")

        if self.case == 0:
            self.solver_direct.opti(sol, self.OCP)
            self.solver_direct.update_solutions(optisol=sol)
            optisol, infos = self.solver_direct.solve()
        else:
            if self.case == 1:
                ocp = OCPCASADI(self.OCP, ode_sol, alg_sol) if params is None else OCPCASADIParams(
                    self.OCP, ode_sol, alg_sol, params
                )
            else:
                ocp = self.OCP

            optisol, infos = self.solver_bvp.solve(sol, ocp)
        return optisol, infos
