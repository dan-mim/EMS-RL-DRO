"""
@daniel.mimouni
This code aim is to propose an implementation of the computation of improving an approximate tree, following the method
proposed in 'Tree approximation for discrete time stochastic process: a process distance approach' from Kovacevic and Pichler.
"""
import numpy as np
import time
from mpi4py import MPI
# my codes
from .MAM import MAM
from .MAM_MPI import MAM_MPI
from .LP_tree_reduction import LP_reduction_nt
from .barycenter_IBP import barycenter_IBP
from scipy.sparse import csc_matrix
from pystocoptim import Atom, get_path_values
from dataclasses import dataclass


__all__ = ["KP_reduction"]


@dataclass
class _MatrixData:
    val: list[float]
    row: list[int]
    col: list[int]


@dataclass
class _TransportDistance:
    pi_cond: _MatrixData
    Dij_t: _MatrixData


def KP_reduction(
        original_filtration: list[list[Atom]], reduced_filtration: list[list[Atom]], method='LP', delta=1000,  itred=7,
        npool=1, lambda_IBP=100, rho=1000, precisionMAM=10**-4) -> tuple[list[list[Atom]], float]:

    assert itred > 0, "itred must be a positive integer"

    assert len(original_filtration) == len(reduced_filtration), (
        "Original and reduced filtrations must have the same number of stages")

    assert len(original_filtration[-1]) > len(reduced_filtration[-1]), (
        "Reduced filtration must have less leaf nodes than the original one")

    start = time.time()
    print("Computing the Kovacevic Pichler's reduction...")

    # name = f'outputs/T{T}_mpi{pool_size}.pkl'
    # with open(name, 'wb') as f:
    #     pickle.dump(RES,f)
    ND_aprx_precedent = 10 ** 5
    Pi = np.zeros((2, 2))

    for niter in range(itred):
        print(f"    Iteration {niter}:")
        print("        Step 1: Optimizing the probabilities")

        # PROBABILITY OPTIMIZATION
        tic = time.time()

        reduced_filtration, ND_aprx, Pi, time_tot = optim_proba(
            original_filtration, reduced_filtration, Pi=Pi, rho=rho, method=method, lambda_IBP=lambda_IBP, npool=npool
        )

        print('              -> optim probab took ', np.round(time.time() - tic,2),f'and the Nested Distance is now {np.round(ND_aprx, 2)}')

        # QUANTIZER OPTIMIZATION
        print("        Step 2: Optimizing the scenario values")
        reduced_filtration = optim_quantizers(original_filtration, reduced_filtration, Pi)

        # Stopping criterion:
        if np.abs(ND_aprx - ND_aprx_precedent) < delta:
            print(f'    The stopping criterion is reached !')
            # name = f'outputs/T{T}_mpi{pool_size}.pkl'
            # with open(name, 'wb') as f:
            #     pickle.dump(dict(G=G, l_tps=l_tps, l_G=l_G, record_t_n=record_t_n, nb_n=nb_n, nb_m=nb_m, ND_aprx=ND_aprx), f)

        ND_aprx_precedent = ND_aprx

    print(f"The scenario tree reduction ({itred} iterations) took {np.round(time.time() - start, 2)}s")

    # Outputs
    return reduced_filtration, ND_aprx


def optim_proba(
        original_filtration, reduced_filtration, Pi=np.zeros((2, 2)), rho=1000, method='LP', lambda_IBP=100, npool=1
):
    comm = MPI.COMM_WORLD
    rank = 0 # comm.Get_rank()
    pool_size = 1 #
    pool_sizeMAM = comm.Get_size()
    """
    :param H: Initial tree
    :param G: Approximated tree structure: only the filtration and the quantifiers are necessary
    H and G must have the same number of stages
    :param Pi: Initialize the transport matrix between two tree
    :param method: Method to compute the barycenter problem : LP, MAM, IBP

    :return:
    * G : an approximated tree of H obtained from a rewritting of 'Tree approximation for discrete time stochastic process:
    a process distance approach' from Kovacevic and Pichler taking advantage of the barycenter LP problem that enables to
    use the Method of Averaged Marginals (MAM) to solve the Linear Program
    * D_ij : the distance matrix between all nodes of H and G. Note that D_ij[0,0] is the nested distance betwen H and G.
    * Pi : the transport matrix between G and H
    * time_tot: total time in second to compute the reduction
    """

    assert method in ['LP', 'MAM', 'IBP']
    # Time management
    start = time.time()

    # PARAMETERS
    # Careful: this parameter can cause a difference more or less significative between the exact distance_GH(H,G) and the
    # distance computed in this algorithm that prevent conditional probabilities to be null.
    # Note that this difference is also due to the approximation of the iterative Pi_hat from the initialization to the end
    epsilon = 10 ** -3

    # TRANSPORT MATRIX AND PROBA OF THE APPROXIMATED TREE
    # I build recursively the transport matrix thanks to the probabilities of the initial tree

    original_nodes_number = sum(len(nodes_stage) for nodes_stage in original_filtration)
    reduced_nodes_number = sum(len(nodes_stage) for nodes_stage in reduced_filtration)

    number_of_stages = len(original_filtration)

    if np.sum(Pi) == 0:
        ## INITIALIZATION OF PI: this is needed because the recursive computation is based on the previous iteration,
        # therefore at iteration 0 it is based on the initialization
        Pi = np.zeros((reduced_nodes_number, original_nodes_number))
        Pi[0, 0] = 1.

        for t in range(number_of_stages - 2, -1, -1):

            for reduced_current_atom in reduced_filtration[t]:
                for original_current_atom in original_filtration[t]:

                    number_of_reduced_children = len(reduced_current_atom.children)
                    # I uniformly fill the transport matrix using the constraint on the conditional proba of the original tree
                    for original_children in original_current_atom.children:
                        for reduced_children in reduced_current_atom.children:
                            # note: p(i|m) = p(i) / p(m)
                            Pi[
                                reduced_children.id, original_children.id
                            ] = original_children.proba / original_current_atom.proba / number_of_reduced_children

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
            val_D_ij.append(
                np.sum(
                    (list_original_values[original_indice] - list_reduced_values[reduced_indice]) ** 2
                )
            )
            row_D_ij.append(reduced_leaf_id)
            col_D_ij.append(original_leaf_id)

    D_ij = csc_matrix((val_D_ij, (row_D_ij, col_D_ij)), (reduced_nodes_number, original_nodes_number))


    ## OPTIMIZATION OF THE PROBABILITIES
    for t in range(number_of_stages - 2, -1, -1):
        starttime = time.time()

        # we go recursively from stage T-1 to stage 0, identifying each time the ancestors of the treated nodes

        original_current_atoms = original_filtration[t]

        reduced_current_atoms = reduced_filtration[t]

        # Parrallel work using MPI:
        outputs = {}
        splitting_work = division_tasks(len(reduced_current_atoms), pool_size)
        for work in splitting_work[rank]:
            reduced_current_atom = reduced_current_atoms[work]
            # Compute the barycenters:
            outputs[reduced_current_atom.id] = _loop_subtree_n(
                original_current_atoms, D_ij, Pi, method, lambda_IBP, rho, epsilon, pool_sizeMAM, reduced_current_atom
            )

        if pool_size > 1:
            l_outputs = comm.gather(outputs, root=0)
            if rank == 0:
                for dico in l_outputs:
                    for key in dico.keys():
                        outputs[key] = dico[key]
            outputs = comm.bcast(outputs, root=0)

        # mise à jour PI avec PI(i, j | m, n) et update Dij
        val_D_ij, row_D_ij, col_D_ij = [], [], []
        for reduced_current_atom in reduced_current_atoms:

            this_pi_cond = outputs[reduced_current_atom.id].pi_cond
            for ind in range(len(this_pi_cond.val)):
                Pi[this_pi_cond.row[ind], this_pi_cond.col[ind]] = this_pi_cond.val[ind]

            this_Dij_t = outputs[reduced_current_atom.id].Dij_t
            val_D_ij += this_Dij_t.val
            row_D_ij += this_Dij_t.row
            col_D_ij += this_Dij_t.col

        D_ij = csc_matrix((val_D_ij, (row_D_ij, col_D_ij)), (reduced_nodes_number, original_nodes_number))


    # Cette methode est équivalente à celle présenté plus bas dans 'Update reduced filtration probabilities' mais peut
    # être un peu plus lente, donc j'ai garé l'autre méthode.
    #     # SET conditional proba at current stage children
    #     original_current_atom0 = original_current_atoms[0]
    #     for reduced_current_atom in reduced_current_atoms:
    #         for reduced_child in reduced_current_atom.children():
    #             # P(j|n) = sum( Pi(m+,j|m,n) )
    #             # note: matrix Pi is the conditional transport plan in the recursive loop
    #             this_proba = np.sum(
    #                 [Pi[reduced_child.id, original_child.id] for original_child in original_current_atom0.children()]
    #             )
    #             reduced_child.set_proba(this_proba)
    #
    # # REBUILD the probability knowing all conditional probabilities
    # for stage in range(1, number_of_stages-1):
    #     for current_reduced_atom in reduced_filtration[stage]:
    #         proba_reduced_atom = current_reduced_atom.proba()
    #         for child_reduced_atom in current_reduced_atom.children():
    #             # P(j) = P(n) * P(j|n)
    #             this_proba = proba_reduced_atom * child_reduced_atom.proba()
    #             child_reduced_atom.set_proba(this_proba)

    # REBUILD the updated Transport matrix
    # Exact method:
    Pi[0, 0] = 1.
    for stage in range(1, number_of_stages):
        for current_original_atom in original_filtration[stage]:
            original_predecessor = current_original_atom.parent
            for current_reduced_atom in reduced_filtration[stage]:
                reduced_predecessor = current_reduced_atom.parent
                Pi[current_reduced_atom.id, current_original_atom.id] *= Pi[
                                                                reduced_predecessor.id, original_predecessor.id
                                                                                                                ]

    # Update reduced filtration probabilities
    total_proba = sum([sum(Pi[reduced_leaf.id, :]) for reduced_leaf in reduced_filtration[-1]])
    for reduced_leaf in reduced_filtration[-1]:
        this_proba = sum(Pi[reduced_leaf.id, :]) / total_proba
        reduced_leaf.proba = this_proba
    for stage in range(number_of_stages - 2, -1, -1):
        for atom_in_stage in reduced_filtration[stage]:
            this_proba = sum([child.proba for child in atom_in_stage.children])
            atom_in_stage.proba = this_proba

    # Time management:
    time_tot = time.time() - start

    # Output
    # D_ij[0,0] is the approached Nested distance
    return reduced_filtration, D_ij[0,0], Pi, time_tot


def _loop_subtree_n(
        current_original_atoms, D_ij, Pi, method, lambda_IBP, rho, epsilon, pool_sizeMAM, current_reduced_atom
):
    """
    Compute the Wasserstein barycenter between subtree at current_reduced_atom and the subtrees at the same stage in the
    original tree.
    :param t: stage number
    :param G: reduced filtration
    :param H: original filtration
    :param nodesG: Ensemble des Atomes de G
    :param nodesH: Ensemble des Atomes de H
    :param list_m: t-Partition de H
    :param D_ij: matrice des distances entre process original et reduit
    :param Pi: Plan de Transport
    :param method: méthode de calcul du WB
    :param lambda_IBP:
    :param rho:
    :param epsilon:
    :param pool_sizeMAM:
    :param n: Reduced Atom
    :return:
    """

    val_D_ij_t = []
    row_D_ij = []
    col_D_ij = []

    val_Pij_cond = []
    row_Pij_cond = []
    col_Pij_cond = []

    b = []  # list of probabilities
    c = {}  # distance matrix with the ponderations
    dist_matrices = []

    # go through each node and treat the subtree (node + its children)
    for indice_original_atom, current_original_atom in enumerate(current_original_atoms):
        # Children of nodes m
        original_atom_children_conditional_probabilities = [
            child.proba / current_original_atom.proba for child in current_original_atom.children
        ]
        reduced_current_atom_children_ids = [child.id for child in current_reduced_atom.children]
        original_current_atom_children_ids = [child.id for child in current_original_atom.children]

        # Family of proba
        b.append( np.array(original_atom_children_conditional_probabilities) )

        # Distance matrix
        dist = D_ij[reduced_current_atom_children_ids, :]
        dist = dist[:, original_current_atom_children_ids]
        dist_matrices.append(dist)
        # ponderation on the distance matrix due to the initialization/previous iteration of Pi
        c[indice_original_atom] = dist * Pi[current_reduced_atom.id, current_original_atom.id]
        c[indice_original_atom] = np.array( c[indice_original_atom].todense() )


    # this provides the barycenter AND the transport matrices !
    if len(current_reduced_atom.children) == 1:
        # this is a trivial case where the subtree of the approximate tree has only one branch (1 child at the node)
        # then directly:
        Pi_k = [np.expand_dims(bi, axis=0) for bi in b]

    # I treat the resolution of the LP as a barycenter problem, using MAM:
    else:
        if method == 'MAM':
            if pool_sizeMAM > 1:
                resMAM = MAM_MPI(b, M_dist=c, exact=False, rho=rho, keep_track=False, computation_time=10,
                             iterations_min=10, iterations_max=200, precision=10 ** -4, logs=False)
            else:
                resMAM = MAM(b, M_dist=c, exact=False, rho=rho, keep_track=False, computation_time=10,
                             iterations_min=10, iterations_max=200, precision=10 ** -4, logs=False)
            Pi_k = resMAM[1]

        elif method == 'IBP':
            # I reshape the probabilities vector as [p1,p2,p(m),0,0,0,...,0], [0,0,0,p1,p2,p(m),0,0,0,...,0] to keep one distance matrix as built in c_
            sumS = sum(len(sublist) for sublist in b)
            acc = 0
            b_ = []
            c_ = []
            for i_m, p in enumerate(b):
                v = np.zeros(sumS)
                v[acc:acc + len(p)] = p
                b_.append(v)
                acc += len(p)
                c_.append(c[i_m])
            c_ = np.concatenate(c_, axis=1)

            resIBP = barycenter_IBP(b_, c_, computation_time=1, iterations_min=10, iterations_max=200,
                                    lambda_sinkhorn=lambda_IBP, precision=10 ** -4)
            Pi_ibp = resIBP[1]
            acc = 0
            Pi_k = []
            for i_m, p in enumerate(b):
                Pi_k.append(Pi_ibp[i_m][:, acc:acc + len(p)])
                acc += len(p)

        elif method == 'LP':
            c_lp = [np.array(c[i_m]) for i_m, _ in enumerate(current_original_atoms)]
            c_lp = np.concatenate(c_lp, axis=1)
            res_LP = LP_reduction_nt(c_lp, b)
            Pi_lp = res_LP[0]
            acc = 0
            Pi_k = []
            for p in b:
                Pi_k.append(Pi_lp[:, acc:acc + len(p)])
                acc += len(p)

    for indice_original_atom, current_original_atom in enumerate(current_original_atoms):
        # 3.3 of the article explains that Pi is not null: it is larger than a small number 'epsilon'
        Pi_k[indice_original_atom] = Pi_k[indice_original_atom].clip(epsilon)
        Pi_k[indice_original_atom] = Pi_k[indice_original_atom] / np.sum(Pi_k[indice_original_atom])  # The sum of all Pi[j,i|n,m] for n and m fixed is equal to 1

        for ind_child_original, child_original_atom in enumerate(current_original_atom.children):
            for ind_child_reduced, child_reduced_atom in enumerate(current_reduced_atom.children):
                val_Pij_cond.append(Pi_k[indice_original_atom][ind_child_reduced, ind_child_original])
                row_Pij_cond.append(child_reduced_atom.id)
                col_Pij_cond.append(child_original_atom.id)

        # FILL the local distance matrix
        val_D_ij_t.append(np.sum(np.multiply(Pi_k[indice_original_atom], dist_matrices[indice_original_atom].todense())))
        row_D_ij.append(current_reduced_atom.id)
        col_D_ij.append(current_original_atom.id)

    pi_cond = _MatrixData(val=val_Pij_cond, row=row_Pij_cond, col=col_Pij_cond)
    Dij_t = _MatrixData(val=val_D_ij_t, row=row_D_ij, col=col_D_ij)
    # Outputs
    return _TransportDistance(pi_cond=pi_cond, Dij_t=Dij_t)


def division_tasks(nb_tasks, pool_size):
    """
    Inputs: (int)
    *nb_tasks
    *pool_size : number of CPU/GPU to divide the tasks between

    Outputs:
    rearranged: numpy list of lists so that rearranged[i] should be treated by CPU[i] (rank=i)
    """
    # The tasks can be equaly divided for each CPUs
    if nb_tasks % pool_size == 0:
        rearranged = np.array([i for i in range(nb_tasks)])
        rearranged = np.split(rearranged, pool_size)

    # Some CPUs will receive more tasks
    else:
        div = nb_tasks // pool_size
        congru = nb_tasks % pool_size
        rearranged1 = np.array([i for i in range(div * congru + congru)])
        rearranged1 = np.split(rearranged1, congru)
        rearranged2 = np.array([i for i in range(div * congru + congru, nb_tasks)])
        rearranged2 = np.split(rearranged2, pool_size - congru)
        rearranged = rearranged1 + rearranged2

    # Output:
    return rearranged


def optim_quantizers(original_filtration, reduced_filtration, Pi):
    """
    This function compute the analytic optimization of the quantizers derived in Theorem 2 (25) in the case of
    Euclidean distance and Wasserstein of order 2 from 'Tree approximation for discrete time stochastic process:
    a process distance approach' from Kovacevic and Pichler
    """
    # number of stages
    number_of_stages = len(original_filtration)
    mat0 = np.zeros_like(original_filtration[0][0].value)
    # we go from stage 0 to stage T
    for t in range(number_of_stages):
        for reduced_current_atom in reduced_filtration[t]:

            num, den = mat0.copy(), mat0.copy()

            for original_current_atom in original_filtration[t]:
                num = num + Pi[reduced_current_atom.id, original_current_atom.id] * original_current_atom.value
                den = den + Pi[reduced_current_atom.id, original_current_atom.id]

            reduced_current_atom.value = num/den

    # Output
    return reduced_filtration




