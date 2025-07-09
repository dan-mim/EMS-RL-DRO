import numpy as np
from numba.typed import List, Dict
from numba import njit, float32, int32
from multiprocessing.pool import Pool
from dataclasses import dataclass
from typing import Union


__all__ = ["reduce_scenarios_forward_selection", ]

@dataclass
class DataReduction:
    scenarios: list[np.ndarray]
    target_number_scenario: int


@dataclass
class ReducedScenarios:
    reduced_scenarios: list[np.ndarray]
    reduced_probas: list[float]


@njit(fastmath=True)
def _compute_pdist_matrix(scenarios, norm=2):
    matdist = np.zeros((len(scenarios), len(scenarios)), dtype=float32)
    m = len(scenarios)
    for i in range(m-1):
        for j in range(i+1, m):
            val = np.sum(np.abs(scenarios[i] - scenarios[j]) ** norm)
            matdist[i, j] = val
            matdist[j, i] = val
    return matdist


@njit(fastmath=True)
def _update_matrix(matdist_k, Jk, Us):
    for k in Jk:
        for u in Jk:
            matdist_k[k, u] = min(matdist_k[k, u], matdist_k[k, Us[-1]])


@njit(fastmath=True)
def _compute_zus(J, matdist_k, m):

    """

    :param J: Set of remaining scenarios
    :param matdist_k: Distances matrices
    :param m: initial number of scenrios
    :return: vector z such that z[i] = mean distance of scenarios of J != i to i
    """
    # Initialization of mean distance vector
    zus = np.zeros((len(J),), dtype=float32)
    # mean distance vector index
    iz = 0

    for u in J:
        # Browsing through remaining scenarios
        rows = np.zeros((len(J)-1,), dtype=int32)
        ik = 0
        # Computation of the mean distance to scenario u
        for i in J:
            if i != u:
                rows[ik] = i
                ik += 1
        # computing the mean distance knowing that scenarios have proba 1 / m
        zus[iz] = np.sum(matdist_k[rows, u]) / m
        iz += 1
    return zus


@njit(fastmath=True)
def _compute_counter(scenarios, Jk, Us):
    counter = Dict()
    for u in Us:
        counter[u] = 1
    for i in Jk:
        best_dist = np.inf
        best_u = -1
        for u in Us:
            this_dist = np.linalg.norm(scenarios[i] - scenarios[u])
            if this_dist < best_dist:
                best_dist = this_dist
                best_u = u
        counter[best_u] += 1
    return counter


@njit(fastmath=True)
def _compute_Jk_Us(R, scenarios):

    matdist_k = _compute_pdist_matrix(scenarios)
    m = len(scenarios)
    zus = _compute_zus(List(range(R)), matdist_k, m)
    u = np.argmin(zus)
    Us = List()
    Us.append(u)
    Jk = List()
    [Jk.append(u) for u in range(m) if u != Us[-1]]

    while len(Us) < R:
        _update_matrix(matdist_k, Jk, Us)

        zus = _compute_zus(Jk, matdist_k, m)

        best_iz = np.argmin(zus)
        Us.append(Jk[best_iz])
        Jknext = List()
        [Jknext.append(u) for u in Jk if u != Us[-1]]
        Jk = Jknext
    return Jk, Us


def _fast_forward_uniform_scenario_selection(R, scenarios):
    m = len(scenarios)
    # Compilation numba
    Jk, Us = _compute_Jk_Us(1, scenarios[:3])
    _ = _compute_counter(scenarios[:3], Jk, Us)

    # start calcul
    Jk, Us = _compute_Jk_Us(R, scenarios)
    counter = _compute_counter(scenarios, Jk, Us)
    reduced_scenarios, reduced_probas = [], []
    for u in Us:
        reduced_scenarios.append(scenarios[u])
        reduced_probas.append(counter[u] / m)
    return reduced_scenarios, reduced_probas


def _forward_selection(original_datas: DataReduction) -> ReducedScenarios:
    """
    This function reduces the number of scenarios using the fast forward selection algorithm from Heitsch and Romisch
    :param original_datas: dict
    :return: ditc
    """
    numba_scenarios = List()
    [numba_scenarios.append(scenario) for scenario in original_datas.scenarios]

    reduced_scens, probas = _fast_forward_uniform_scenario_selection(
            original_datas.target_number_scenario, numba_scenarios
        )

    return ReducedScenarios(reduced_scenarios=reduced_scens, reduced_probas=probas)


def reduce_scenarios_forward_selection(scenarios_list: Union[list[list[np.ndarray]],list[np.ndarray]],
                                       nscens_red: Union[list[int], int],
                                       pool_number=-1) \
        -> tuple[Union[list[list[np.ndarray]],list[np.ndarray]], Union[list[int], int]]:
    """

    :param scenarios_list: is a list of list of len the dimension of the scenario.
    :param nscens_red: is a list of list of len the dimension of the scenario. But if nscens_red is an INT instead of a LIST
    then scenarios_list is automatically understood as 1D. and the output will be given in the same framework.
    :param pool_number
    :return:
    """
    scenario_are_1D = False
    if isinstance(nscens_red, int):
        scenarios_list = [scenarios_list]
        nscens_red = [nscens_red]
        scenario_are_1D = True


    list_data_reduction = [
        DataReduction(scenarios_list[i_rand_var], nscens_red[i_rand_var])
        for i_rand_var in range(len(scenarios_list))
    ]

    if pool_number > 1:
        with Pool(pool_number) as pool:
            list_reduced_scens = pool.map(_forward_selection, list_data_reduction)

    else:
        list_reduced_scens = [_forward_selection(original_datas) for original_datas in list_data_reduction]

    list_reduced_scenarios = [
        data_red.reduced_scenarios for data_red in list_reduced_scens
    ]

    list_probas = [
        data_red.reduced_probas for data_red in list_reduced_scens
    ]

    #Output:
    if scenario_are_1D:
        list_reduced_scenarios, list_probas = list_reduced_scenarios[0], list_probas[0]
    return list_reduced_scenarios, list_probas




