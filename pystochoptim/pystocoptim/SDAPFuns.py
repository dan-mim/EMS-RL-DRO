import numpy as np
from pyopticontrol import OptiSol, OptiControl, IndirectOCP
from .FiltrationFuns import Atom, compute_filtration
from typing import Optional, Union
from scipy.interpolate import interp1d
import cvxpy as cp
from numpy import float64
from numpy.typing import NDArray
import time as tm
from .UtilFunctions import ScenarioData, SDAPData, ipms_socp, non_anticipative_projection
from functools import partial
from multiprocessing import Pool


__all__ = ["SOCSDAP", "SDAPSolver"]


class SOCSDAP(IndirectOCP):

    def __init__(self, dim_control: int, dim_state: int, eps_ref: float):
        self.dim_control = dim_control
        self.dim_state = dim_state
        self.fun_scenario = interp1d(np.array([0., 1.]), np.array([1., 2.]))
        self.fun_v_proj = interp1d(np.array([0., 1.]), np.array([1., 2.]))
        self.rho_proj = 0.
        self.eps_ref = eps_ref
        self.eps = eps_ref

    def set_scenario_data(self, scenario_data: ScenarioData):
        self.fun_scenario = interp1d(scenario_data.optisol.time, scenario_data.scenario)
        if scenario_data.sdap is not None:
            self.fun_v_proj = interp1d(scenario_data.optisol.time,
                                       2 * scenario_data.control - scenario_data.sdap.z_control_sdap)
            self.rho_proj = 2 * scenario_data.sdap.u_sdap - scenario_data.sdap.zu_sdap
        self.eps = scenario_data.eps

    def eval_fs(self,
                time: NDArray[float64],
                ode_sol: NDArray[float64],
                alg_sol: NDArray[float64]) -> float:
        pass


class SDAPSolver:

    def __init__(self,
                 socp: SOCSDAP,
                 epi_ocp: SOCSDAP,
                 options: Optional[dict] = None,
                 theta: Optional[float] = -1.,
                 r_proj_ambiguity_set: Optional[float] = 1.,
                 print_data: Optional[bool] = True,
                 tol_keep_proba: Optional[float] = 1e-2):
        self.socp = socp
        self.opticontrol = OptiControl(socp, options=options)
        self.epi_ocp = epi_ocp
        options = dict(exact_prb=False, display=0, check_jacobian=False)
        self.opticontrol_epi = OptiControl(epi_ocp, options=options)
        self.theta = theta # DRO radius parameter
        self.r_sdap = r_proj_ambiguity_set
        self.time_init = np.empty((1,))
        self.ode_sol_init = np.empty((1, 1))
        self.alg_sol_init = np.empty((1, 1))
        self.ode_sol_epi_init = np.empty((1, 1))
        self.alg_sol_epi_init = np.empty((1, 1))
        self.params_init = np.zeros((2,))
        self.print_data = print_data
        self.tol_keep_proba = tol_keep_proba
        self.ambiguity_set_projection = None
        self.mu = None
        self.mu0 = None
        self.transport_plan = None

    def solve(self,
              scenarios_reduced_tree: list[NDArray[float64]],
              probas_reduced_tree: list[float],
              scenarios_original_tree: list[NDArray[float64]],
              probas_original_tree: list[float],
              delay_hazard: float,
              time: NDArray[float64],
              ode_sol: NDArray[float64],
              alg_sol: NDArray[float64],
              ode_sol_epi: NDArray[float64],
              alg_sol_epi: NDArray[float64],
              tol_control: float,
              tol_sdap: float,
              iteration_max: int,
              tol_clust=1e-6,
              parallelize_sdap: Optional[int] = 0,
              pool_number: Optional[int] = -1,
              decay_ratio_ipm: Optional[float] = .1,
              verbose: Optional[bool] = False,
              max_qp_iter: Optional[int] = 1000000,
              solver_qp: Optional[str] = 'CLARABEL'):

        # Check PHA parallelization setting
        assert parallelize_sdap in [0, 1, 2], ("parallelize_sdap must be in {0, 1, 2}")

        # Check that the non-anticipative delay is non negative
        assert delay_hazard >= 0., "Delay hazard must be non negative"

        # Define parallelization strategy from parallelize_sdap
        if parallelize_sdap == 0:
            # In this cas no computation is conducted in parallel
            parallel_sdap_init, parallelize_sdap_iter = False, False
        elif parallelize_sdap == 1:
            # In this case, only the initialization of the sdap is conducted in parallel
            parallel_sdap_init, parallelize_sdap_iter = True, False
        else:
            # In this cas, both the initialization and the iteration of the SDAP are done in parallel
            parallel_sdap_init, parallelize_sdap_iter = True, True

        total_computation_time = tm.time()

        print("Solving SDAP ...")
        print("       Step 1 : Compute filtration")
        filtration_reduced_tree, reduced_tree_scenario_datas, time, ode_sol, alg_sol, ode_sol_epi, alg_sol_epi, number_of_steps_to_drop = (
            self._compute_filtration_scenario_datas(scenarios_reduced_tree, probas_reduced_tree, delay_hazard, time,
                                                    ode_sol, alg_sol, ode_sol_epi, alg_sol_epi, tol_clust)
        )
        self.time_init, self.ode_sol_init, self.alg_sol_init, self.ode_sol_epi_init, self.alg_sol_epi_init = (
            time, ode_sol, alg_sol, ode_sol_epi, alg_sol_epi
        )

        print("       Step 2 : Initialize SDAP ")
        # Compute some parameters: theta is the ambiguity set radius, it is kept only if no theta is provided
        # in the initialization; d_matrix is the matrix distance between reduced tree and original tree in term of scenario values

        theta_min, self.d_matrix = initialize_theta_dmatrix(
            scenarios_original_tree, probas_original_tree, scenarios_reduced_tree, number_of_steps_to_drop
        )
        # if self.theta == -1:
        #     self.theta = theta_min
        #     print(f"The radius of the ambiguity set (theta) has been set to the smallest possible value: theta = {np.round(theta_min, 2)}.")
        # assert self.theta >= theta_min, f"The radius of the ambiguity set (theta) has to be greater than {np.round(theta_min, 2)}."
        if self.theta <= theta_min:
            self.theta = theta_min
            print(f"The radius of the ambiguity set (theta) has to be greater than {np.round(theta_min, 2)}.")
            print(
                f"The radius of the ambiguity set (theta) has been set to the smallest possible value: theta = {np.round(theta_min, 2)}.")

        # Update optisol to make it now solve the projection onto the epigraph problem.
        tic = tm.time()

        # Initialize SDAP variables by solving the anticipative problem
        self._update_controls(reduced_tree_scenario_datas,
                              decay_ratio_ipm=decay_ratio_ipm,
                              parallelize_sdap_init=parallel_sdap_init,
                              pool_number=pool_number)

        for scenario_data in reduced_tree_scenario_datas:
            z_control_sdap = scenario_data.control.copy()
            self.socp.set_scenario_data(scenario_data)
            zu_sdap = self.socp.eval_fs(scenario_data.optisol.time,
                                        scenario_data.optisol.ode_sol,
                                        scenario_data.optisol.alg_sol)
            zv_sdap = zu_sdap
            sdap_data = SDAPData(z_control_sdap=z_control_sdap, zu_sdap=zu_sdap, zv_sdap=zv_sdap)
            scenario_data.sdap = sdap_data
            scenario_data.eps = self.socp.eps_ref
            scenario_data.warm_start = False
            # define optisol per scenario using the initialization to prepare for the projection onto the epigraph optimal
            # control solving
            optisol = OptiSol(time=self.time_init,
                              ode_sol=self.ode_sol_epi_init.copy(),
                              alg_sol=self.alg_sol_epi_init.copy(),
                              params=self.params_init.copy()
                              )
            scenario_data.optisol = optisol

        self._build_ambiguity_set_projection_problem(
            np.ones((len(reduced_tree_scenario_datas),)),
            probas_original_tree,
            scenarios_original_tree,
            scenarios_reduced_tree
        )
        print(f"              -> computed in {np.round(tm.time() - tic, 2)}s")

        print("       Step 3 : Optimization loop:")
        success = False
        niter = 1

        while not success:

            self._sdap_iteration(reduced_tree_scenario_datas, filtration_reduced_tree, decay_ratio_ipm,
                                 parallelize_sdap_iter=parallelize_sdap_iter, pool_number=pool_number, verbose=verbose,
                                 max_qp_iter=max_qp_iter, solver_qp=solver_qp)

            print("              Step 3.6 : Evaluate fixed point stopping criteria")
            error_control, error_u_sdap, error_v_sdap = self._compute_errors(reduced_tree_scenario_datas)
            print("                 ||control_hat^{k+1} - control^{k}|| = ", error_control)
            print("                 ||u_hat^{k+1} - u^{k}|| = ", error_u_sdap)
            print("                 ||v_hat^{k+1} - u^{k}|| = ", error_v_sdap)
            if (error_control < tol_control and error_u_sdap < tol_sdap and error_v_sdap < tol_sdap) or niter >= iteration_max:
                success = True
            print(f"                Iteration {niter} is successfully computed")
            niter += 1

        # Outputs
        controls = []
        states = []
        reduced_proba = []
        new_scenarios = []
        obj_value = 0.
        tol_keep_proba = self.tol_keep_proba / len(reduced_tree_scenario_datas)
        indice_proba_greater_than_one = -1
        ind = 0
        for scenario_data in reduced_tree_scenario_datas:
            if scenario_data.sdap.proba_scen > tol_keep_proba:
                new_scenarios.append(scenario_data.scenario)
                controls.append(scenario_data.control)
                states.append(scenario_data.optisol.ode_sol[:self.epi_ocp.dim_state])
                obj_value += self.epi_ocp.eval_fs(
                    scenario_data.optisol.time, scenario_data.optisol.ode_sol, scenario_data.optisol.alg_sol
                ) * scenario_data.sdap.proba_scen
                reduced_proba.append(scenario_data.sdap.proba_scen)
                if scenario_data.sdap.proba_scen >= 1.:
                    indice_proba_greater_than_one = ind
                ind += 1

        if indice_proba_greater_than_one >= 0:
            new_scenarios = [new_scenarios[indice_proba_greater_than_one]]
            controls = [controls[indice_proba_greater_than_one]]
            states = [states[indice_proba_greater_than_one]]
            reduced_proba = [reduced_proba[indice_proba_greater_than_one]]

        filtration_reduced_tree = compute_filtration(
            new_scenarios, tol=tol_clust, probas=reduced_proba, check_proba=False
        )

        print(f"The SDAP algorithm has been computed in {np.round(tm.time() - total_computation_time, 2)}s")
        return (reduced_tree_scenario_datas[0].optisol.time, controls, states, obj_value, filtration_reduced_tree,
                reduced_proba, new_scenarios)

    def _compute_filtration_scenario_datas(self,
                                           scenarios: list[NDArray[float64]],
                                           probas: list[float],
                                           delay_hazard: float,
                                           time: NDArray[float64],
                                           ode_sol: NDArray[float64],
                                           alg_sol: NDArray[float64],
                                           ode_sol_epi: NDArray[float64],
                                           alg_sol_epi: NDArray[float64],
                                           tol_clust=1e-6) -> (
            tuple[list[list[Atom]],
            list[ScenarioData],
            NDArray[float64],
            NDArray[float64],
            NDArray[float64],
            NDArray[float64],
            NDArray[float64],
            int]):

        filtration = compute_filtration(scenarios, tol=tol_clust, probas=probas)

        # Compute delayed filtration for optimal control
        number_of_steps_to_drop = 0
        if delay_hazard > 0.:
            # Compute the number of steps to drop
            number_of_steps_to_drop = sum(np.where(time < time[0] + delay_hazard, 1, 0))
            # Drop the last stages of the filtration
            filtration = filtration[:-number_of_steps_to_drop]
            # Drop the first stages of the scenarios
            for i in range(len(scenarios)):
                scenarios[i] = scenarios[i][:, number_of_steps_to_drop:]
            # Drop the first stages of the solutions of the OCPs
            time, ode_sol, alg_sol, ode_sol_epi, alg_sol_epi = (
                time[number_of_steps_to_drop:], ode_sol[:, number_of_steps_to_drop:],
                alg_sol[:, number_of_steps_to_drop:], ode_sol_epi[:, number_of_steps_to_drop:],
                alg_sol_epi[:, number_of_steps_to_drop:]
            )

        assert (len(time) == scenarios[0].shape[1]), ("time array does not match scenario's length")

        scenario_datas = []

        for ind, scenario in enumerate(scenarios):

            optisol = OptiSol(time=time, ode_sol=ode_sol.copy(), alg_sol=alg_sol.copy())

            scenario_data = ScenarioData(
                indice=ind,
                scenario=scenario,
                eps=self.socp.eps_ref,
                warm_start=False,
                optisol=optisol,
                control=optisol.alg_sol[:self.socp.dim_control, :])

            scenario_datas.append(scenario_data)

        return filtration, scenario_datas, time, ode_sol, alg_sol, ode_sol_epi, alg_sol_epi, number_of_steps_to_drop

    def _update_controls(self,
                         scenario_datas: list[ScenarioData],
                         decay_ratio_ipm: float,
                         parallelize_sdap_init: bool = False,
                         pool_number: int = -1):
        """
        Solve the optimal control problem for each scenario
        """
        if parallelize_sdap_init and pool_number > 1:
            ipms = partial(ipms_socp,
                           self.opticontrol,
                           self.socp.eps_ref,
                           self.time_init,
                           self.ode_sol_init,
                           self.alg_sol_init,
                           None,
                           self.opticontrol.solver_bvp.display,
                           decay_ratio_ipm,
                           self.print_data)

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
                optimize_scenario_sdap(self.opticontrol,
                                       self.socp.eps_ref,
                                       self.time_init,
                                       self.ode_sol_init,
                                       self.alg_sol_init,
                                       None,
                                       self.opticontrol.solver_bvp.display,
                                       decay_ratio_ipm,
                                       self.print_data,
                                       scenario_data,
                                       epi_projection=False
                                       )

    def _build_ambiguity_set_projection_problem(self,
                                                mu0: NDArray[float64],
                                                probas_original_tree: list[float],
                                                scenarios_original_tree: list[NDArray],
                                                scenarios_reduced_tree: list[NDArray]):
        l_scenarios_original_tree = [sc[:, 1:] for sc in scenarios_original_tree]

        S = mu0.shape[0]

        S_original_tree = len(l_scenarios_original_tree)

        # Initialization
        self.mu = cp.Variable(S, nonneg=True)
        self.mu0 = cp.Parameter(S)
        self.transport_plan = cp.Variable((S, S_original_tree), nonneg=True)

        # Objectif : minimization of ||mu - mu0||^2
        objective = cp.Minimize(cp.sum_squares(self.mu - self.mu0))

        # Constraints
        constraints = [
            cp.sum(cp.multiply(self.transport_plan, self.d_matrix)) <= self.theta,
            cp.sum(self.transport_plan, axis=0) == np.array(probas_original_tree),
            cp.sum(self.transport_plan, axis=1) == self.mu,
            cp.sum(self.transport_plan) == 1
        ]

        self.ambiguity_set_projection = cp.Problem(objective, constraints)

    def proj_ambiguity_set(self, mu0: NDArray[float64], verbose: bool, max_qp_iter: int, solver_qp: str):
        """
        Projection onto the Wasserstein ambiguity set:
        The center of the ambiguity set is the original tree,
        The structure of the scenario is the structure of the reduced tree.
        """
        self.mu0.value = mu0
        self.ambiguity_set_projection.solve(max_iter=max_qp_iter, warm_start=True, verbose=verbose,
                                            solver=solver_qp)

        assert self.ambiguity_set_projection.status == "optimal", "Projection on the ambiguity set has not converged"

        return self.mu.value

    def _update_u_sdap(self, reduced_tree_scenario_datas: list[ScenarioData]):
        for scenario_data in reduced_tree_scenario_datas:
            scenario_data.sdap.u_sdap = (scenario_data.sdap.zu_sdap + scenario_data.sdap.zv_sdap) / 2

    def _update_v(self,
                  reduced_tree_scenario_datas: list[ScenarioData],
                  verbose: bool,
                  max_qp_iter: int,
                  solver_qp: str):
        """
        Compute the prox that actually boils down to a projection onto the ambiguity set (here I use proj_ambiguity_set
        to project onto the Wasserstein ambiguity set)
        """
        u_sdap_minus_zv_sdap = np.empty((len(reduced_tree_scenario_datas), ), dtype=float64)
        for i, sd in enumerate(reduced_tree_scenario_datas):
            u_sdap_minus_zv_sdap[i] = 2 * sd.sdap.u_sdap - sd.sdap.zv_sdap

        projection = self.proj_ambiguity_set(self.r_sdap * u_sdap_minus_zv_sdap, verbose, max_qp_iter, solver_qp)
        v_hat_sdap = (u_sdap_minus_zv_sdap - 1. / self.r_sdap * projection)
        for i, scenario_data in enumerate(reduced_tree_scenario_datas):
            scenario_data.sdap.v_hat_sdap = v_hat_sdap[i]
            scenario_data.sdap.proba_scen = float(projection[i])

    def _update_x_hat_u_hat(self,
                            reduced_tree_scenario_datas: list[ScenarioData],
                            decay_ratio_ipm: float,
                            parallelize_sdap: bool,
                            pool_number: int = -1):
        """
        Solve the projection onto the epigraph problem
        """
        if parallelize_sdap and pool_number > 1:
            ipms = partial(
                ipms_socp,
                self.opticontrol_epi,
                self.epi_ocp.eps_ref,
                self.time_init,
                self.ode_sol_epi_init,
                self.alg_sol_epi_init,
                self.params_init,
                self.opticontrol.solver_bvp.display,
                decay_ratio_ipm,
                self.print_data
            )
            with Pool(pool_number) as pool:
                list_eps_bvpsols = pool.map(ipms, reduced_tree_scenario_datas)

            for ind_scen, scenario_data in enumerate(reduced_tree_scenario_datas):
                this_eps, bvpsol = list_eps_bvpsols[ind_scen]
                scenario_data.warm_start = True
                scenario_data.eps = this_eps
                scenario_data.optisol = bvpsol
                scenario_data.sdap.control_hat_sdap = bvpsol.alg_sol[:self.opticontrol_epi.OCP.dim_control]
                scenario_data.sdap.u_hat_sdap = bvpsol.params[0]
        else:
            for scenario_data in reduced_tree_scenario_datas:

                optimize_scenario_sdap(self.opticontrol_epi,
                                       self.epi_ocp.eps_ref,
                                       self.time_init,
                                       self.ode_sol_epi_init,
                                       self.alg_sol_epi_init,
                                       self.params_init,
                                       self.opticontrol.solver_bvp.display,
                                       decay_ratio_ipm,
                                       self.print_data,
                                       scenario_data,
                                       epi_projection=True
                                       )
    @staticmethod
    def _compute_errors(reduced_tree_scenario_datas: list[ScenarioData]) -> tuple[float, float, float]:

        error_control = sum(np.sum((sd.sdap.control_hat_sdap - sd.control) ** 2) * sd.sdap.proba_scen
                            for sd in reduced_tree_scenario_datas
                            )

        norm_control = sum(np.sum(sd.control ** 2) * sd.sdap.proba_scen for sd in reduced_tree_scenario_datas)

        error_u_sdap = sum(np.sum((sd.sdap.u_hat_sdap - sd.sdap.u_sdap) ** 2) * sd.sdap.proba_scen
                           for sd in reduced_tree_scenario_datas
                           )

        norm_u_sdap = sum(
            np.sum(sd.sdap.u_sdap ** 2) * sd.sdap.proba_scen for sd in reduced_tree_scenario_datas
        )

        error_v_sdap = sum(np.sum((sd.sdap.v_hat_sdap - sd.sdap.u_sdap) ** 2) * sd.sdap.proba_scen
                           for sd in reduced_tree_scenario_datas
                           )

        return error_control / norm_control, error_u_sdap / norm_u_sdap, error_v_sdap / norm_u_sdap

    def _sdap_iteration(self, reduced_tree_scenario_datas: list[ScenarioData],
                        filtration_reduced_tree: list[list[Atom]],
                        decay_ratio_ipms: float,
                        parallelize_sdap_iter: bool,
                        pool_number: int,
                        verbose: bool,
                        max_qp_iter: int,
                        solver_qp: str):

        print("              Step 3.1 : Project onto non anticipative constraint : update control")
        z_controls_sdap = [sd.sdap.z_control_sdap for sd in reduced_tree_scenario_datas]
        na_z_controls_sap = non_anticipative_projection(z_controls_sdap, filtration_reduced_tree)
        for iscen, scenario_data in enumerate(reduced_tree_scenario_datas):
            scenario_data.control = na_z_controls_sap[iscen]

        print("              Step 3.2 : Update u")
        self._update_u_sdap(reduced_tree_scenario_datas)

        print("              Step 3.3 : Project onto ambiguity set - update v_hat")
        # Compute the prox of v -> max_p E_p(v) over r.
        tic = tm.time()
        self._update_v(reduced_tree_scenario_datas, verbose, max_qp_iter, solver_qp)
        print(f"                       -> computed in {np.round(tm.time() - tic, 2)}s")

        print("              Step 3.4 : Project onto the epigraph - update control_hat and u_hat")
        tic = tm.time()
        self._update_x_hat_u_hat(reduced_tree_scenario_datas,
                                 decay_ratio_ipm=decay_ratio_ipms,
                                 parallelize_sdap=parallelize_sdap_iter,
                                 pool_number=pool_number)
        print(f"                       -> computed in {np.round(tm.time() - tic, 2)}s")

        print("              Step 3.5 : Update zx, zu and zv")
        for scenario_data in reduced_tree_scenario_datas:
            scenario_data.sdap.z_control_sdap = (
                    scenario_data.sdap.z_control_sdap + scenario_data.sdap.control_hat_sdap - scenario_data.control
            )
            scenario_data.sdap.zu_sdap = scenario_data.sdap.zu_sdap + scenario_data.sdap.u_hat_sdap - scenario_data.sdap.u_sdap
            scenario_data.sdap.zv_sdap = scenario_data.sdap.zv_sdap + scenario_data.sdap.v_hat_sdap - scenario_data.sdap.u_sdap


def optimize_scenario_sdap(solver: OptiControl,
                           eps_ref: float,
                           time_init: NDArray[float64],
                           ode_sol_init: NDArray[float64],
                           alg_sol_init: NDArray[float64],
                           params_init: Union[NDArray[float64], None],
                           display: int,
                           decay_ratio: float,
                           print_data: bool,
                           scenario_data: ScenarioData,
                           epi_projection: bool):

    this_eps, bvpsol = ipms_socp(solver, eps_ref, time_init, ode_sol_init, alg_sol_init, params_init, display,
                                 decay_ratio, print_data, scenario_data)
    scenario_data.warm_start = True
    scenario_data.eps = this_eps
    scenario_data.optisol = bvpsol
    if not epi_projection:
        scenario_data.control = bvpsol.alg_sol[:solver.OCP.dim_control]
    else:
        scenario_data.sdap.control_hat_sdap = bvpsol.alg_sol[:solver.OCP.dim_control]
        scenario_data.sdap.u_hat_sdap = bvpsol.params[0]


def initialize_theta_dmatrix(scenarios_original_tree: list[NDArray[float64]],
                             probas_original_tree: list[float],
                             scenarios_reduced_tree: list[NDArray[float64]],
                             number_of_steps_to_drop: int=0) -> tuple[float, NDArray[float64]]:
    """
    d_matrix is the matrix distance between the original tree and the reduced tree in term of scenario values
    since it does not change through the SDAP algorithm we can compute it only once beforehand.
    This code also provides an initial guess for the radius of the ambiguity set theta.
    number_of_steps_to_drop is due to the delay hazard
    """
    l_scenarios_original_tree = [sc[:, number_of_steps_to_drop:] for sc in scenarios_original_tree]
    S = len(scenarios_reduced_tree)
    S_original_tree = len(l_scenarios_original_tree)
    d_matrix = np.zeros(
        (S, S_original_tree))  # cdist(scenarios_reduced_tree, scenarios_original_tree, metric='sqeuclidean')
    for i in range(S):
        for j in range(S_original_tree):
            d_matrix[i, j] = np.sum((scenarios_reduced_tree[i] - l_scenarios_original_tree[j]) ** 2)

    # the minimum admissible radius for the ambiguity set is :
    theta_min = np.sum( probas_original_tree @ np.min(d_matrix, axis=0) )
    # mu = np.ones(S) / S  # proba_reduced_tree
    # transport_plan = np.zeros((S, S_original_tree))
    # for s in range(S):
    #     for j in range(S_original_tree):
    #         transport_plan[s, j] = mu[s] * probas_original_tree[j]
    #
    # theta = np.sum(np.multiply(d_matrix, transport_plan))

    return theta_min, d_matrix
