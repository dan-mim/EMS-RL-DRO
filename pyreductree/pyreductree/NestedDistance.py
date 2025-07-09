# RECURSIVE COMPUTATION : SOLVE THE LINEAR PROGRAM
from scipy.sparse import csc_matrix
import scipy
from scipy import optimize
import time
import numpy as np
from pystocoptim import Atom, get_path_values


__all__ = ["nested_distance"]


# COMPUTE DISTANCE BETWEEN TWO TREES:
def nested_distance(original_filtration: list[list[Atom]], reduced_filtration: list[list[Atom]]) -> tuple[float, float]:
    """
    This function compute the distance between the trees G and H, as derived in (17) from
    'Tree approximation for discrete time stochastic process: a process distance approach' from Kovacevic and Pichler
    :param original_filtration: original filtration as a list of list of Atoms
    :param reduced_filtration: reduced filtration as a list of list of Atoms
    :return: D_ij[0,0], D_ij, Time : The nested distance between the two trees is {D_ij[0,0]}, computerd in {Time} s
    """
    # Manage Time
    start = time.time()
    original_nodes_number = sum([len(l) for l in original_filtration])
    reduced_nodes_number = sum([len(l) for l in reduced_filtration])
    number_of_stages = len(original_filtration)

    # INITAILIZATION OF THE DISTANCE MATRIX : exact process distance between the leaves
    # Distance matrix initialization
    val_D_ij = []
    row_D_ij = []
    col_D_ij = []

    list_orginal_leaf_id, list_original_values = [], []
    for original_leaf in original_filtration[-1]:
        original_values = get_path_values(original_leaf)
        list_orginal_leaf_id.append(original_leaf.id)
        list_original_values.append(original_values)

    list_reduced_leaf_id, list_reduced_values = [], []
    for reduced_leaf in reduced_filtration[-1]:
        reduced_values = get_path_values(reduced_leaf)
        list_reduced_leaf_id.append(reduced_leaf.id)
        list_reduced_values.append(reduced_values)

    for original_indice, original_leaf_id in enumerate(list_orginal_leaf_id):
        for reduced_indice, reduced_leaf_id in enumerate(list_reduced_leaf_id):
            val_D_ij.append(np.linalg.norm(
                np.array(list_original_values[original_indice]) - np.array(list_reduced_values[reduced_indice])
            ) ** 2
                            )
            row_D_ij.append(reduced_leaf_id)
            col_D_ij.append(original_leaf_id)

    D_ij = csc_matrix((val_D_ij, (row_D_ij, col_D_ij)), (reduced_nodes_number, original_nodes_number))

    for t in range(number_of_stages - 2, -1, -1):
        # we go recursively from stage T-1 to stage 0, identifying each time the ancestors of the treated nodes

        original_current_atoms = original_filtration[t]

        reduced_current_atoms = reduced_filtration[t]

        for indice_original_atom, current_original_atom in enumerate(original_current_atoms):
            for indice_reduced_atom, current_reduced_atom in enumerate(reduced_current_atoms):
                # Conditional Probabilities
                original_atom_children_conditional_probabilities = np.array([
                    child.proba / current_original_atom.proba for child in current_original_atom.children
                ])
                reduced_atom_children_conditional_probabilities = np.array([
                    child.proba / current_reduced_atom.proba for child in current_reduced_atom.children
                ])
                reduced_current_atom_children_ids = [child.id for child in current_reduced_atom.children]
                original_current_atom_children_ids = [child.id for child in current_original_atom.children]
                # Distance matrix
                dist = D_ij[reduced_current_atom_children_ids, :]
                dist = dist[:, original_current_atom_children_ids]
                dist = np.array( dist.todense() )
                # Compute distance
                dist_max = np.max(dist)
                original_atom_children_conditional_probabilities = np.abs(original_atom_children_conditional_probabilities)/np.sum(original_atom_children_conditional_probabilities)
                reduced_atom_children_conditional_probabilities = np.abs(reduced_atom_children_conditional_probabilities)/np.sum(reduced_atom_children_conditional_probabilities)
                res = LP_dlr_mn(
                    dist / dist_max, reduced_atom_children_conditional_probabilities, original_atom_children_conditional_probabilities
                                )
                if type(res[0]) != float:
                    res = LP_dlr_mn(
                        np.round(dist / dist_max, 3), original_atom_children_conditional_probabilities, reduced_atom_children_conditional_probabilities
                                    )
                # Recursivity is saved in D_ij
                D_ij[current_reduced_atom.id, current_original_atom.id] = res[0] * dist_max

    # Output
    Time = time.time() - start
    return (D_ij[0, 0], Time)


def LP_dlr_mn(c,p,q):
    """
    This function is a linear program using the method High to compute (17) from the paper
    'Tree approximation for discrete time stochastic process: a process distance approach' from Kovacevic and Pichler
    Its aim is to recursively compute the distance between two trees.
    :param c: Distance matrix between i_s and j_s (R x S)
    :param p: constraint 1 P(i|m)
    :param q: constraint 2 P'(j|n)
    :return:
    Distance_exact = dlr(m,n),
    Pi_exact = pi(.,. | m,n)
    Time = time of execution
    """
    # trivial case
    # if len(p)==len(q) and np.sum([np.abs(p[i]-q[i]) for i in range(len(p))])==0:
    #     return(0,0)
    # Time management:
    start = time.time()

    # Objective function
    R, S = c.shape
    c = np.reshape(c, (R * S,))

    # Building A: Constraint (right sum of Pi to get p)
    shape_A = (R, R * S)
    indices_values_A = np.ones(R * S)
    # lines indices:
    line_indices_A = np.array([])
    for r in range(R):
        line_indices_A = np.append(line_indices_A, np.ones(S) * r)
    # column indices:
    column_indices_A = np.linspace(0, R * S - 1, R * S)

    # Building B: Constraint (left sum of Pi to get q)
    shape_B = (S, R*S)
    indices_values_B = np.ones(R*S)
    # lines indices:
    line_indices_B = np.array([])
    for s in range(S):
        line_indices_B = np.append(line_indices_B, np.ones(R) * s)
    # column indices: time_spent for R_=S_=1600
    column_indices_B = np.array([])
    for s in range(S):
        column_indices_B = np.append(column_indices_B, np.linspace(0,S*(R-1),R) + s)

    # Building A_eq = concatenate((A, B), axis=0)
    shape_Aeq = (R + S, R * S)
    indices_values_Aeq = np.append(indices_values_A, indices_values_B)
    line_indices_Aeq = np.append(line_indices_A, line_indices_B + R)
    column_indices_Aeq = np.append(column_indices_A, column_indices_B)
    A_eq = csc_matrix((indices_values_Aeq, (line_indices_Aeq, column_indices_Aeq)), shape_Aeq)

    # b_eq:
    b_eq = np.append(p, q)

    # Resolution: with the bound x >= 0
    AEQ =  csc_matrix.todense(A_eq)
    res = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0,None), method='highs')  #, bounds=(0,None)

    # computing time:
    end = time.time()
    Time = np.round((end - start), 2)

    # Output:
    # Optimization terminated successfully.
    Distance_exact = res.fun
    # Pi_exact = res.x
    # Pi_exact = np.reshape(Pi_exact, (R, S))

    return(Distance_exact, Time)  #Pi_exact