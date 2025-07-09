import numpy as np
from pyopticontrol import OptiSol, OptiControl, IndirectOCP
from .FiltrationFuns import Atom, compute_filtration
from typing import Optional, Union
from scipy.interpolate import interp1d
from functools import partial
from multiprocessing.pool import Pool
from numpy.typing import NDArray
from numpy import float64
from .UtilFunctions import ScenarioData, RPHAData, ipms_socp, non_anticipative_projection

__all__ = ["RPHASolver", "SOCRPHA"]


class SOCRPHA(IndirectOCP):
    """
    This class contains the optimal control problem
    to solve.
    """

    def __init__(
            self, pha_weight: float, variance_weight: float, dim_control: int, dim_state: int, eps_ref: float):
        """

        :param pha_weight:
        :param variance_weight:
        :param dim_control:
        :param dim_state:
        :param eps_ref:
        """
        self.pha_weight = pha_weight
        self.variance_weight = variance_weight
        self.dim_control = dim_control
        self.dim_state = dim_state
        self.fun_scenario = interp1d(np.array([0., 1.]), np.array([1., 2.]))
        self.fun_z_pha = interp1d(np.array([0., 1.]), np.array([1., 2.]))
        self.fun_mult_pha = interp1d(np.array([0., 1.]), np.array([1., 2.]))
        self.eps_ref = eps_ref
        self.eps = eps_ref

    def set_scenario_data(self, scenario_data: ScenarioData):
        self.fun_scenario = interp1d(scenario_data.optisol.time, scenario_data.scenario)
        if scenario_data.rpha is not None:
            self.fun_z_pha = interp1d(scenario_data.optisol.time, scenario_data.rpha.z_pha)
            self.fun_mult_pha = interp1d(scenario_data.optisol.time, scenario_data.rpha.mult_pha)
        self.eps = scenario_data.eps


class RPHASolver:

    def __init__(self, socp: SOCRPHA, options: Optional[dict] = None, print_data: Optional[bool] = True,
                 tasks_by_worker: Optional[int] = 50):
        self.socp = socp
        self.opticontrol = OptiControl(socp, options=options)
        self.time_init = np.empty((1,))
        self.ode_sol_init = np.empty((1, 1))
        self.alg_sol_init = np.empty((1, 1))
        self.print_data = print_data
        self._tasks_by_worker = tasks_by_worker

    def solve(self,
              scenarios: list[NDArray[float64]],
              probas: list[float],
              delay_hazard: float,
              time: NDArray[float64],
              ode_sol: NDArray[float64],
              alg_sol: NDArray[float64],
              tol_pha: Optional[float] = 1e-5,
              tol_clust: Optional[float] = .01,
              parallelize_pha: Optional[int] = 0,
              pool_number: Optional[int] = -1,
              decay_ratio_ipms: Optional[float] = .1) \
            -> tuple[NDArray[float64], list[NDArray[float64]], list[NDArray[float64]], list[list[Atom]]]:

        # Check PHA parallelization setting
        assert parallelize_pha in [0, 1, 2], ("parallelize_pha must be in {0, 1, 2}")

        # Check that the non-anticipative delay is non negative
        assert delay_hazard >= 0., "Delay hazard must be non negative"

        # Define parallelization strategy from parallelize_pha
        if parallelize_pha == 0:
            # In this cas no computation is conducted in parallel
            parallel_z_pha_init, parallel_pha_iter = False, False
        elif parallelize_pha == 1:
            # In this case, only the initialization of the z_pha's is conducted in parallel
            parallel_z_pha_init, parallel_pha_iter = True, False
        else:
            # In this cas, both the initialization and the iteration of the RPHA are done in parallel
            parallel_z_pha_init, parallel_pha_iter = True, True

        if self.print_data:
            print("       Step 1 : Compute filtration")

        # Compute filtration from scenario data and measurement delay
        filtration, scenario_datas, time_init, ode_sol_init, alg_sol_init = self._compute_filtration_scenario_datas(
            scenarios, probas, delay_hazard, time, ode_sol, alg_sol, tol_clust
        )
        self.time_init, self.ode_sol_init, self.alg_sol_init = (
            time_init.copy(), ode_sol_init.copy(), alg_sol_init.copy()
        )
        # Compute optimal solutions of deterministic problems with quadratic term without non anticipative constraints
        if self.print_data:
            print("       Step 2 : Compute anticipative solutions ")

        # Set the pha weight to zero to solve the optimal control problems
        pha_weight = self.socp.pha_weight
        self.socp.pha_weight = 0.

        # Solve the optimal control problems
        self._update_controls_rpha(scenario_datas, parallel_z_pha_init, pool_number, decay_ratio_ipms)

        # Compute non anticipative projection of the optimal controls
        na_controls = non_anticipative_projection(
            [scenario_data.control for scenario_data in scenario_datas], filtration, probas
        )

        # Initialize z_pha variables for each scenario
        for i in range(len(scenario_datas)):
            scenario_datas[i].rpha.z_pha = na_controls[i]

        # Restaure pha_weight to prescribed value
        self.socp.pha_weight = pha_weight

        if self.print_data:
            print("       Step 3 : Start fixed-point iterations ")
        success = False
        niter = 0
        previous_controls, previous_z_phas = None, None

        # Compute pool_number so that each worker computes at least self._tasks_by_worker problems
        this_pool_number = min(pool_number, round(len(scenario_datas) / self._tasks_by_worker))

        while not success:

            # Perform a RPHA iteration
            self._pha_iteration(scenario_datas, filtration, probas, parallel_pha_iter, this_pool_number, decay_ratio_ipms)

            if self.print_data:
                print("          iter #",  str(niter))

            # Compute fixed-point errors
            error_control, error_z_phas, previous_controls, previous_z_phas = self._compute_errors_rpha(scenario_datas,
                                                                                                        probas,
                                                                                                        previous_controls,
                                                                                                        previous_z_phas)

            if self.print_data:
                print("            ||u^{k+1} - u^{k}|| = ", error_control)
                print("            ||z^{k+1} - z^{k}|| = ", error_z_phas)

            # Convergence test
            success = max(error_control, error_z_phas) <= tol_pha
            niter += 1

        # Retrieve optimal controls and states variables
        controls = [sd.control for sd in scenario_datas]
        states = [sd.optisol.ode_sol[:self.socp.dim_state] for sd in scenario_datas]

        return scenario_datas[0].optisol.time, controls, states, filtration

    def _compute_filtration_scenario_datas(self,
                                           scenarios: list[NDArray[float64]],
                                           probas: list[float],
                                           delay_hazard: float,
                                           time: NDArray[float64],
                                           ode_sol: NDArray[float64],
                                           alg_sol: NDArray[float64],
                                           tol_clust=.01) -> (
            tuple[list[list[Atom]], list[ScenarioData], NDArray[float64], NDArray[float64], NDArray[float64]]):
        """
        This method computes the SOCP's filtration according the scenarios and the delay of measurement. In addition,
        this function computes the list of ScenarioData corresponding to each scenario
        :param scenarios: List of scenarios as NumPy arrays
        :param probas: List of floats such that probas[i] is the probability of scenarios[i]
        :param delay_hazard: Delay for computing the filtration
        :param time: Numpy array with time steps for optimal control
        :param ode_sol: Initial solution of the ODEs of the SOCPs as NumPy array
        :param alg_sol: Initial solution of the AEs of the SOCPs as Numpy array
        :param tol_clust: positive float for clustering scenarios in filtration computation
        :return: filtration as a list[list[Atom]] and the list of ScenarioData
        """
        # Compute the filtration generated by the scenarios
        filtration = compute_filtration(scenarios, tol=tol_clust, probas=probas)

        # Compute delayed filtration for optimal control
        if delay_hazard > 0.:
            # Compute the number of steps to drop
            number_of_steps_to_drop = sum(np.where(time < time[0] + delay_hazard, 1, 0))
            # Drop the last stages of the filtration
            filtration = filtration[:-number_of_steps_to_drop]
            # Drop the first stages of the scenarios
            for i in range(len(scenarios)):
                scenarios[i] = scenarios[i][:, number_of_steps_to_drop:]
            # Drop the first stages of the solutions of the OCPs
            time, ode_sol, alg_sol = (
                time[number_of_steps_to_drop:], ode_sol[:, number_of_steps_to_drop:],
                alg_sol[:, number_of_steps_to_drop:]
            )

        assert (len(time) == scenarios[0].shape[1]), ("time array does not match scenario's length")

        # Compute ScenarioDatas
        scenario_datas = []
        for ind, scenario in enumerate(scenarios):

            optisol = OptiSol(time=time, ode_sol=ode_sol.copy(), alg_sol=alg_sol.copy())
            rphadata = RPHAData(z_pha=np.zeros((self.socp.dim_control, len(time))),
                                mult_pha=np.zeros((self.socp.dim_control, len(time)))
                                )

            scenario_data = ScenarioData(
                indice=ind, scenario=scenario, eps=self.socp.eps_ref, warm_start=False, optisol=optisol,
                rpha=rphadata)
            scenario_datas.append(scenario_data)

        return filtration, scenario_datas, time, ode_sol, alg_sol

    def _pha_iteration(self,
                       scenario_datas: list[ScenarioData],
                       filtration: list[list[Atom]],
                       probas: list[float],
                       parallelize_pha: bool,
                       pool_number: int,
                       decay_ratio_ipms: float):
        """
        This method performs an iteration of the RPHA
        :param scenario_datas: List of ScenarioData
        :param filtration: Filtration that the optimal control must generate
        :param probas: Scenarios probabilities
        :param parallelize_pha: Boolean for parallel computing of the deterministic problems
        :param pool_number: Number of workers for parallelization
        :param decay_ratio_ipms: float valued in (0, 1).
        :return:
        """
        # Compute u^{k+1} from z^{k} and lambda^{k}
        self._update_controls_rpha(scenario_datas, parallelize_pha, pool_number, decay_ratio_ipms)
        # Compute z^{k+1} from u^{k+1}
        self._update_z_rpha(scenario_datas, filtration, probas)
        # Compute lambda^{k+1} from u^{k+1}
        self._update_mult_rpha(scenario_datas, filtration, probas)

    def _update_controls_rpha(self,
                              scenario_datas: list[ScenarioData],
                              parallelize_pha: bool,
                              pool_number: int,
                              decay_ratio_ipms: float):

        if parallelize_pha and pool_number > 1:
            ipms = partial(
                ipms_socp, self.opticontrol, self.socp.eps_ref, self.time_init, self.ode_sol_init,
                self.alg_sol_init, None, self.opticontrol.solver_bvp.display, decay_ratio_ipms, self.print_data
            )
            with Pool(pool_number) as pool:
                list_eps_bvpsols = pool.map(ipms, scenario_datas)

            for ind_scen, scenario_data in enumerate(scenario_datas):
                this_eps, bvpsol = list_eps_bvpsols[ind_scen]
                scenario_data.warm_start = True
                scenario_data.eps = this_eps
                scenario_data.optisol = bvpsol
                scenario_data.control = bvpsol.alg_sol[:self.socp.dim_control]

        else:
            for scenario_data in scenario_datas:
                optimize_scenario_rpha(self.opticontrol, self.socp.eps_ref, self.time_init, self.ode_sol_init,
                                       self.alg_sol_init, None, self.opticontrol.solver_bvp.display, decay_ratio_ipms,
                                       self.print_data, scenario_data)

    def _update_mult_rpha(self,
                          scenario_datas: list[ScenarioData],
                          filtration: list[list[Atom]],
                          probas: Optional[list[float]] = None):

        controls = [scenario_data.control for scenario_data in scenario_datas]
        non_anticipative_controls = non_anticipative_projection(controls, filtration, probas)
        for scen_number, scenario_data in enumerate(scenario_datas):
            scenario_data.rpha.mult_pha = scenario_data.rpha.mult_pha + self.socp.pha_weight * (
                    scenario_data.control - non_anticipative_controls[scen_number]
            )

    def _update_z_rpha(self,
                       scenario_datas: list[ScenarioData],
                       filtration: list[list[Atom]],
                       probas: Optional[list[float]] = None):

        arg = [2. * sd.control - sd.rpha.z_pha for sd in scenario_datas]
        exp_arg = sum(arg[i] * probas[i] for i in range(len(arg)))

        if self.socp.variance_weight == np.inf:
            for scen in scenario_datas:
                scen.rpha.z_pha = scen.rpha.z_pha - scen.control + exp_arg
        else:
            non_anticipative_arg = non_anticipative_projection(arg, filtration, probas)
            for iscen, scen in enumerate(scenario_datas):
                scen.rpha.z_pha = (
                        scen.rpha.z_pha - scen.control
                        + 1. / (self.socp.pha_weight + self.socp.variance_weight) * (
                                self.socp.variance_weight * exp_arg + self.socp.pha_weight * non_anticipative_arg[iscen]
                        )
                )
                
    @staticmethod
    def _compute_errors_rpha(
            scenario_datas: list[ScenarioData],
            probas: list[float],
            previous_controls: Optional[list[NDArray[float64]]] = None,
            previous_z_phas: Optional[list[NDArray[float64]]] = None
    ) -> tuple[float, float, list[NDArray[float64]], list[NDArray[float64]]]:

        ns = len(scenario_datas)

        if previous_controls is not None and previous_z_phas is not None:

            control_error = sum(
                np.sum((scenario_datas[ind].control - previous_controls[ind]) ** 2) * probas[ind] for ind in range(ns)
            )

            norm_control = sum(
                np.sum(scenario_datas[ind].control ** 2) * probas[ind] for ind in range(ns)
            )

            z_pha_error = sum(
                np.sum((scenario_datas[ind].rpha.z_pha - previous_z_phas[ind]) ** 2) * probas[ind] for ind in range(ns)
            )

            norm_z_pha = sum(
                np.sum(scenario_datas[ind].rpha.z_pha ** 2) * probas[ind] for ind in range(ns)
            )

            norm_control_error = control_error / norm_control

            norm_z_pha_error = z_pha_error / norm_z_pha
        else:
            norm_control_error = np.inf

            norm_z_pha_error = np.inf

        previous_controls = [scenario_datas[ind].control for ind in range(ns)]

        previous_z_phas = [scenario_datas[ind].rpha.z_pha for ind in range(ns)]

        return norm_control_error, norm_z_pha_error, previous_controls, previous_z_phas


def optimize_scenario_rpha(solver: OptiControl,
                           eps_ref: float,
                           time_init: NDArray[float64],
                           ode_sol_init: NDArray[float64],
                           alg_sol_init: NDArray[float64],
                           params_init: Union[NDArray[float64], None],
                           display: int,
                           decay_ratio: float,
                           print_data: bool,
                           scenario_data: ScenarioData):

    this_eps, bvpsol = ipms_socp(solver, eps_ref, time_init, ode_sol_init, alg_sol_init, params_init, display,
                                 decay_ratio, print_data, scenario_data)

    scenario_data.warm_start = True
    scenario_data.eps = this_eps
    scenario_data.optisol = bvpsol
    scenario_data.control = bvpsol.alg_sol[:solver.OCP.dim_control]




