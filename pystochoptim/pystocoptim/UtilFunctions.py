from numpy import float64
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional, Union
from pyopticontrol import OptiSol, OptiControl
from .FiltrationFuns import Atom

__all__ = ["RPHAData", "SDAPData", "ScenarioData", "ipms_socp", "non_anticipative_projection"]


@dataclass
class RPHAData:
    z_pha: NDArray[float64]
    mult_pha: NDArray[float64]


@dataclass
class SDAPData:
    z_control_sdap: NDArray[float64]
    zu_sdap: float
    zv_sdap: float
    u_sdap: Optional[float] = None
    v_hat_sdap: Optional[float] = None
    u_hat_sdap: Optional[float] = None
    control_hat_sdap: Optional[NDArray[float64]] = None
    proba_scen: Optional[float] = None


@dataclass
class ScenarioData:
    # Scenario indice
    indice: int
    # Scenario value
    scenario: NDArray[float64]
    # Value of IPM's weight for solving the OCP with data from this class
    eps: Optional[float] = 0.
    # Authorize Warm-start for solving the OCP by IPMs. If True, the IPMs is solved using eps as initial weighting
    # parameter
    warm_start: Optional[bool] = False
    # OptiSol corresponding of the OCP's optimal solution
    optisol: Optional[OptiSol] = None
    # Value of the z value from the RPHA
    rpha: Optional[RPHAData] = None
    sdap: Optional[SDAPData] = None
    # Value of the optimal control, i.e. the solution of the RPHA for the scenario of this Scenario data
    control: Optional[NDArray[float64]] = None


def non_anticipative_projection(xs: list[NDArray[float64]],
                                filtration: list[list[Atom]],
                                probas: Optional[list[float]] = None) -> list[NDArray[float64]]:
    is_proba = probas is not None and len(probas) == len(xs)
    na_xs = xs.copy()
    for stage, stage_partition in enumerate(filtration):

        if len(stage_partition) >= len(xs):
            break

        for atom in stage_partition:
            if is_proba:
                mean_pbat_atom = sum(xs[scen_id][:, stage] * probas[scen_id] for scen_id in atom.scen_ids) / atom.proba
            else:
                mean_pbat_atom = sum(xs[scen_id][:, stage] for scen_id in atom.scen_ids) / len(atom.scen_ids)

            for scen_id in atom.scen_ids:
                na_xs[scen_id][:, stage] = mean_pbat_atom

    return na_xs


def ipms_socp(solver: OptiControl,
              eps_ref: float,
              time_init: NDArray[float64],
              ode_sol_init: NDArray[float64],
              alg_sol_init: NDArray[float64],
              params_init: Union[NDArray[float64], None],
              display: int,
              decay_ratio: float,
              print_data: bool,
              scenario_data: ScenarioData) -> tuple[float, OptiSol]:

    solver.OCP.set_scenario_data(scenario_data)

    this_eps, this_warm_start = scenario_data.eps, scenario_data.warm_start

    if this_warm_start and scenario_data.optisol.ode_sol is not None and scenario_data.optisol.alg_sol is not None:
        # In this case we initialize the problem with optimal solution from previous RPHA step with small
        # IPM weight. We also do a defensive copy of the solution to ensure no error is made when computing
        # the stopping criteria for the fixed point iterations of the RPHA
        solver.OCP.eps = this_eps

        time, ode_sol, alg_sol = (
            scenario_data.optisol.time, scenario_data.optisol.ode_sol.copy(), scenario_data.optisol.alg_sol.copy()
        )
        alg_sol_mid = scenario_data.optisol.alg_sol_mid.copy() if scenario_data.optisol.alg_sol_mid else None
        params = scenario_data.optisol.params.copy() if scenario_data.optisol.params is not None else None

    else:
        # In this case, no warm-start is performed and we start from initial guess
        this_eps = eps_ref
        solver.OCP.eps = this_eps
        time, ode_sol, alg_sol = time_init.copy(), ode_sol_init.copy(), alg_sol_init.copy()
        alg_sol_mid = None
        params = params_init if params_init is not None else None

    convergence = False
    tol = 1e-6
    alpha0 = decay_ratio
    alpha = alpha0
    bvpsol = OptiSol(time=time, ode_sol=ode_sol, alg_sol=alg_sol, alg_sol_mid=alg_sol_mid, params=params)
    niter = 0
    while not convergence:
        if display > 0 and print_data:
            print("i = ", scenario_data.indice)
            print('eps = ', this_eps)

        bvpsol_new, infos = solver.solve(bvpsol)

        if not infos.success:
            if this_warm_start:
                if print_data:
                    print("      failed warm-start at indice = ", scenario_data.indice)
                    print('      eps = ', this_eps)
                this_eps = eps_ref
                solver.OCP.eps = this_eps
                time, ode_sol, alg_sol = time_init.copy(), ode_sol_init.copy(), alg_sol_init.copy()
                params = params_init if params_init is not None else None
                bvpsol = OptiSol(time=time, ode_sol=ode_sol, alg_sol=alg_sol, params=params)
                this_warm_start = False
            elif alpha <= .95:
                if print_data:
                    print("      No warm start ; start over optimization with smaller decrease")
                    print('      eps = ', this_eps)
                    print('      alpha = ', alpha)
                    print("      indice = ", scenario_data.indice)
                this_eps /= alpha
                alpha = .5 * alpha + .5
            else:
                if print_data:
                    print("alpha = ", alpha)
                    print("scenario indice = ", scenario_data.indice)
                    print('eps = ', this_eps)
                    print("niter = ", niter)
                    print("infos.NLSE_infos = ", infos.NLSE_infos)
                raise Exception("Failure optimization")
        else:
            bvpsol = bvpsol_new
            alpha = alpha0
        convergence = this_eps <= tol
        if not convergence:
            this_eps *= alpha
            solver.OCP.eps = this_eps
        niter += 1
    return this_eps, bvpsol