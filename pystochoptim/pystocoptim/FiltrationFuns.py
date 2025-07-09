import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional
from numpy import float64, int64
from numpy.typing import NDArray

__all__ = ["Atom", "compute_filtration", "get_path_values", "retrieve_scenarios_from_filtration_bfs",
           "retrieve_scenarios_from_filtration_dfs"]


@dataclass
class Atom:
    scen_ids: set
    depth: int
    id: int
    parent: Optional['Atom'] = None
    children: Optional[list['Atom']] = None
    value: Optional[NDArray[float64]] = None
    proba: Optional[float] = 0.
    ind: Optional[int] = None # this is an identification number that can be stored temporarily


def compute_filtration(predicted_scenarios: list[NDArray[float64]],
                       tol: Optional[float] = 1e-6,
                       probas: Optional[list] = None,
                       check_proba: Optional[bool] = True) -> list[list[Atom]]:
    """
    :param predicted_scenarios: List of scenarios, each scenario is a n x Times array where n is the dimension of the random process
    :param quantiles: List of quantiles. The list has n elements n being the dimension of the random process
    :param tol: tolerance for clustering
    :param probas: Optional : List of scenario's probability
    :param check_proba: In the case where probas is given, we check if probas is indeed a probability
    :return: filtration as a list of list of atoms
    """
    # Test if the list probas is indeed a probability
    if probas is not None and check_proba:
        assert len(probas) == len(predicted_scenarios) and (np.abs(sum(probas) - 1.) <= 1e-8 and min(probas) >= -1e-8),\
            ("Scenarios do not form a probabilty: probas = " + str(probas))
    else:
        inv_nscen = 1. / len(predicted_scenarios)
        probas = [inv_nscen for _ in range(len(predicted_scenarios))]

    # The root id is 0
    atom_id = 0
    # Initialize root value to the mean value of scenarios at stage 0
    val_root = sum(predicted_scenarios[i][:, 0] * probas[i] for i in range(len(predicted_scenarios)))
    # Root atom is a set containing all scenario ids
    scen_ids = {i for i in range(len(predicted_scenarios))}

    # Build root Atom
    root_atom = Atom(scen_ids=scen_ids, depth=0, id=atom_id)
    root_atom.value = val_root
    root_atom.proba = sum(probas)

    # Initialize the filtration as a list whose first element is a list containing only the root Atom
    filtration = [[root_atom]]

    # Traversing of the time-steps to compute the t-Partitions of the set of scenarios
    for t_ind in range(predicted_scenarios[0].shape[1] - 1):

        # Initialize the list of Atom at stage t_ind corresponding to the (t_ind+1)-partition of the set of scenario
        atoms_at_next_step = []

        # Now let us go through all atoms at this stage and compute its _children, i.e. the scenarios from this atom
        # that become distinct at next time-step.
        for atom in filtration[t_ind]:

            atom_id += 1
            children_atoms, atom_id = _compute_this_atom_children(t_ind, predicted_scenarios, atom, tol, probas, atom_id)

            # link the children to the parent atom
            atom.children = children_atoms
            atoms_at_next_step += children_atoms

        # Add t-partition represented by a list of atom to the t-partition
        filtration.append(atoms_at_next_step)

    return filtration


def _compute_this_atom_children(t_ind: int,
                                predicted_scenarios: list[NDArray[float64]],
                                this_atom: Atom,
                                tol: float,
                                probas: list[float],
                                atom_id: int) -> tuple[list[Atom], int]:
    """
    :param t_ind:  Current time indice
    :param predicted_scenarios: 2D array whose rows are the scenarios for the stochastic process
    :param this_atom: Atom at t_ind such that all scenarios in this atom match the actual realization up to t_ind
    :param tol: Error tolerance on relative distances of scenarios values
    :param probas: List of probability of each scenario
    :return: this_atom _children
    """
    # This atom's _children list instanciation
    children_atoms = []

    # Initialization of the set of remaining scenarios as the set of elements in current atom
    remaining_scenarios_ids = this_atom.scen_ids.copy()

    # While loop to scatter the scenarios in the atom's _children
    while len(remaining_scenarios_ids) > 0:
        # Get a scenario from remaining set
        this_scen_id = next(iter(remaining_scenarios_ids))

        # Create a set from elements in remaining_scenarios close enough to this_scen
        new_atom_scen_ids = {this_scen_id}

        # Get the value of this_scen at stage t_ind + 1
        this_next_value = predicted_scenarios[this_scen_id][:, t_ind + 1]

        # Compute expected value of new_atom with respect to the conditional probability of the scenarios of the
        # atom
        mean_next_atom_value = this_next_value * probas[this_scen_id]

        # Compute new_atom probability which is the sum of probability of the elements of this atom
        proba_atom = probas[this_scen_id]

        # Going through all remaining_scenarios
        for that_scen_id in remaining_scenarios_ids:
            if that_scen_id == this_scen_id:
                continue

            # Get value of that_scen at stage t_ind + 1
            that_next_value = predicted_scenarios[that_scen_id][:, t_ind + 1]

            # Check if that_next_value is close enough to this_next_value so that that_scenario belongs to
            # next_atom
            if np.linalg.norm(that_next_value - this_next_value) / np.linalg.norm(this_next_value) <= tol:
                new_atom_scen_ids.add(that_scen_id)
                # keep computing expected value of the atom
                mean_next_atom_value += that_next_value * probas[that_scen_id]
                # and of its probability
                proba_atom += probas[that_scen_id]

        # Create new atom from new_atom_elements and set its depth equal to the depth of this_atom + 1
        new_atom = Atom(scen_ids=new_atom_scen_ids, depth=this_atom.depth + 1, id=atom_id)
        atom_id += 1

        # Update value of new_atom
        new_atom.value = mean_next_atom_value / proba_atom

        # Update parent of new_atom
        new_atom.parent = this_atom

        # Update proba of new_atom
        new_atom.proba = proba_atom

        # Add new_atom to the list of _children
        children_atoms.append(new_atom)

        # Remove scenarios in new_atom from the set of remaining scenarios ids
        remaining_scenarios_ids = remaining_scenarios_ids.difference(new_atom.scen_ids)

    return children_atoms, atom_id-1


def retrieve_scenarios_from_filtration_bfs(filtration: list[list[Atom]]) -> tuple[list[NDArray[float64]], list[float]]:
    root = filtration[0][0]
    shape_scenario = (len(root.value), len(filtration))
    scenario_numbers = len(root.scen_ids)
    list_scenarios = [np.zeros(shape_scenario) for _ in range(scenario_numbers)]
    probas = [0. for _ in range(scenario_numbers)]
    explored = set()
    queue = deque([root])
    while len(queue) > 0:
        atom = queue.popleft()
        explored.add(atom)
        for scen_id in atom.scen_ids:
            list_scenarios[scen_id][:, atom.depth] = atom.value

        if atom.children is not None:
            for atom_child in atom.children:
                if atom_child not in explored:
                    queue.append(atom_child)
        else:
            scen_id = next(iter(atom.scen_ids))
            probas[scen_id] = atom.proba

    return list_scenarios, probas


def retrieve_scenarios_from_filtration_dfs(filtration: list[list[Atom]]) -> tuple[list[NDArray[float64]], list[float]]:
    """
    This function retrieves scenarios and their probability from the filtration in argument by a _depth-first-search like
    algorithm
    :param filtration: As a List of List of Atom
    :return: List of Scenarios and their probability
    """
    # The filtration being a tree, its root is the first element of the first list
    root = filtration[0][0]

    # Computing the shape of the scenario, i.e. the size of the random vector and the number of stages
    shape_scenario = (len(root.value), len(filtration))

    # Initialiazing the list of scenario and their probabilities
    list_scenarios, probas = [], []

    # Initializing the set of Atoms already explored
    explored = {root.id}

    # Initialize the value of the current scenario
    this_scenario = []

    # Start recursive DFS to update list_scenarios and probas
    _dfs_filtration(root, explored, list_scenarios, probas, this_scenario, shape_scenario)

    return list_scenarios, probas


def _dfs_filtration(atom: Atom,
                    explored: set[int],
                    list_scenarios: list[NDArray[float64]],
                    probas: list[float],
                    this_scenario: NDArray[float64],
                    shape_scenario: tuple[int, int]):
    """

    :param atom: Atom being explored
    :param explored: Set of atoms already explored
    :param list_scenarios: List of scenario to update
    :param probas: Corresponding probabilities
    :param this_scenario: Current scenario being computed while traversing the tree _depth-first
    :param shape_scenario: Shape of the scenario
    """

    # If the current scenario is of maximum length before adding atom's value, this mean the DFS algorithm has
    # explored a full path from root to leave and that we are now exploring the latest unexplored fork of that path
    # This is why we initialize this-scenario as having the same atom._depth first values as the last complete scenario
    if len(this_scenario) == shape_scenario[1]:
        this_scenario = [list_scenarios[-1][:, i].reshape((shape_scenario[0], 1)) for i in range(atom.depth)]

    # Add atom's value to this-scenario
    this_scenario.append(atom.value.reshape((shape_scenario[0], 1)))

    # Recursively explore _children. If there are none, then we are at a leaf node and we can update both the list
    # of scenario by adding this scenario and the list of probability which is the probability of this atom.
    if atom.children is not None:
        for child_atom in atom.children:
            if child_atom.id not in explored:
                explored.add(child_atom.id)
                _dfs_filtration(child_atom, explored, list_scenarios, probas, this_scenario, shape_scenario)
    else:
        probas.append(atom.proba)
        list_scenarios.append(np.concatenate(this_scenario, axis=1))


def _get_previous_values(atom: Atom, list_values: list[NDArray[float64]]):
    if atom.parent is not None:
        list_values.append(atom.parent.value)
        _get_previous_values(atom.parent, list_values)


def get_path_values(leaf: Atom) -> NDArray[float64]:
    list_values = [leaf.value]
    _get_previous_values(leaf, list_values)
    return np.vstack(list_values).transpose()


if __name__ == "__main__":
    from timeit import default_timer as dt
    nscen = 1000
    len_scen = 200
    scenarios = []
    probas = []
    for scen in range(nscen):
        vals = np.zeros((2, len_scen))
        for t in range(len_scen):
            rdn = np.random.rand()
            if rdn > 0.5:
                vals[0, t] = 1.
            else:
                vals[0, t] = -1.
            rdn = np.random.rand()
            if rdn > 0.5:
                vals[1, t] = 2.
            else:
                vals[1, t] = -2.

        scenarios.append(vals)
        probas.append(1. / nscen)
    t0 = dt()
    filtration = compute_filtration(scenarios, tol=1e-6, probas=probas)
    t1 = dt()

    print("t1 - t0 = ", t1 - t0)

    leaf_node = filtration[-1][0]
    path_values = get_path_values(leaf_node)
    x = np.array(path_values)
    v1 = np.linalg.norm(x, axis=1) ** 2
    v2 = np.sum(v1)
    v0 = np.linalg.norm(x) ** 2
    print("a = ", 1)




