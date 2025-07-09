# -*- coding: utf-8 -*-
"""
Inspired from Operator_splitting_low_memory but with parallel workers
Inspired from MAM_balanced
It is to the date 13/03/2024 the best MAM version I developped

@author: mimounid
"""

# Imports

# Basics
import numpy as np
import time
import matplotlib.pyplot as plt
# Import parallel work management:
from mpi4py import MPI
import sys
import pickle
import scipy
from scipy import spatial


__all__ = ["MAM"]


# Vectorize function that project vectors onto a simplex, inspired by https://gist.github.com/mblondel/c99e575a5207c76a99d714e8c6e08e89
def projection_simplex(V, z=1, axis=None):
    """
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    axis: None or int
        axis=None: project V by P(V.ravel(); z)
        axis=1: project each V[i] by P(V[i]; z[i])
        axis=0: project each V[:, j] by P(V[:, j]; z[j])
    """
    if axis == 1:
        n_features = V.shape[1]
        U = np.sort(V, axis=1)[:, ::-1]
        z = np.ones(len(V)) * z
        cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
        ind = np.arange(n_features) + 1
        cond = U - cssv / ind > 0
        rho = np.count_nonzero(cond, axis=1)
        theta = cssv[np.arange(len(V)), rho - 1] / rho
        return np.maximum(V - theta[:, np.newaxis], 0)

    elif axis == 0:
        return projection_simplex(V.T, z, axis=1).T

    else:
        V = V.ravel().reshape(1, -1)
        return projection_simplex(V, z, axis=1).ravel()


# Computing the distance matrix:
def distance_matrix(M, S, exact=False):
    """
    This function compute the distance matrix for the exact (free support) problem
    of the inexact problem (fixed support)
    M is the number of considered densities
    S is the size of the probability densities support
    #(R is the size of the barycenter support we look for)
    """
    ### Ancienne methode:
    if not exact:
        R = S
        # eps_R and eps_S are the discretization size of the barycenter image (for R) and the sample images (for S)
        eps_R = 1 / R ** .5
        eps_S = 1 / S ** .5

        la = np.linspace(0, R - 1, R)
        lb = np.linspace(0, S - 1, S)

        x_R, y_R = la % R ** .5 * eps_R + eps_R / 2, la // R ** .5 * eps_R + eps_R / 2
        x_S, y_S = lb % S ** .5 * eps_S + eps_S / 2, lb // S ** .5 * eps_S + eps_S / 2

        XA = np.array([x_R, y_R]).T
        XB = np.array([x_S, y_S]).T
        M_dist = spatial.distance.cdist(XA, XB, metric='euclidean')

    if exact:
        # Method that respect Bogward theorems:
        K = int(np.round(S ** .5))
        X, Y = np.meshgrid(np.append(np.arange(0, K - 1, 1 / M), K - 1), np.append(np.arange(0, K - 1, 1 / M), K - 1))
        ptsK = np.column_stack((X.ravel(), Y.ravel()))

        X, Y = np.meshgrid(np.linspace(0, K - 1, K), np.linspace(0, K - 1, K))
        ptsk = np.column_stack((X.ravel(), Y.ravel()))

        # Calcul de la distance
        M_dist = spatial.distance.cdist(ptsK, ptsk)

    return M_dist ** 2


def build_M_dist(rank, splitting_work, pool_size, comm, R, S, b, exact=False):
    """
    The distance matrix is build in rank 0 and then it is split with the other processors
    but only the interesting parts of the matrix are shared.
    Then the original distance matrix is delate.
    """
    if rank == 0:
        print('Computing distance matrix...')
        sys.stdout.flush()
        # only one processor build the large distance matrix
        M = len(b)
        M_dist = distance_matrix(M, S, exact) / S  # .astype(np.float16)
        # share the distance matrices between the processors:
        # share_M_btw_procs(M_dist, b, R, splitting_work)
        Mat_dist = {}
        S = {}
        # Sharing the matrix distance to each ranks (but only with the needed columns)
        for rg in range(1, pool_size):
            for m in splitting_work[rg]:
                I = b[m] > 0
                S[m] = np.sum(I)
                Mat_dist[m] = np.reshape(M_dist[:, I], (R * S[m],))
            # sending the matrix
            comm.send(S, dest=rg)
            comm.send(Mat_dist, dest=rg)
            S = {}
            Mat_dist = {}
        # same but direct for rank 0
        for m in splitting_work[0]:
            I = b[m] > 0
            S[m] = np.sum(I)
            Mat_dist[m] = M_dist[:, I]
        # delate the large M_dist
        del M_dist

    # receiving the matrices
    if rank != 0:
        S = comm.recv(source=0)
        Mat_dist = comm.recv(source=0)
        print(rank, S, splitting_work[rank], Mat_dist.keys())
        sys.stdout.flush()
        for m in splitting_work[rank]:
            print(m, S[m])
            sys.stdout.flush()
            Mat_dist[m] = np.reshape(Mat_dist[m], (R, S[m]))

    # Output
    return Mat_dist


########## MAM algorithm ############
# Optimized Operator Splitting method
def MAM(b, M_dist=[], rho=1000, keep_track=True, evry_it=10, name='MAM.pkl', visualize=False, exact=False,
        computation_time=100, iterations_min=3, iterations_max=1000, precision=10 ** -4, logs=True, rank=0):
    """
    This version of MAM adresses pb with very large but sparse support. In this case the distance matrix are computed one 
    by one (instead of building a huge distance matrix and only using the sparse rare columns)
    Input:
    *b: (resolution x n) vector collection of n probability distribution.
    *M_dist:( resolution x resolution) is the distance matrix between every supports.
    /!\ Note that M_dist can be a dictionary where M_dist[m] is the distance matrix between the support barycenter and the measure m.
    *exact: (bool) if True we use the free support method (large support considered, refer to Borckwardt), else fixed support, 
    meaning we keep the same support than have the b[i]s for the barycenter.
    *rho: (float) is an hyperparameter that is linked to the convergence speed but has no impact on the solution
    if rho is precised in the arguments of the function then this rho will be used
    * gamma: (float) value of the unbalanced parameter, if gamma=0, MAM treats the balanced barycenter problem
    * keep_track: (bool) should we keep records of the past barycenter approximations and other datas.
    * evry_it: (int) the datas are saved every evry_it iterations
    * name: (str) name of the saved data in the current repository
    * visualize: (bool) if True an approximation of the barycenter is displayed (usefull if image datas are treated)
    * computation_time: (float) maximal computation time in seconds
    * iterations_min: (int) minimal number of iterations
    * iterations_max: (int) maximal number of iterations
    * precision: (float) stopping criteria based on the fixed point method, using the norm max

    Output: p, P, Time, Wdist, l_precisionB, Precision, Iterations, total_time, iterations_k
    * p: (R x 1) the probability distribution of the calculated barycenter
    * P: (R x iterations_k) matrix that store 'p_' after eache iteration of the algorithm
    * Time: (1 x iterations_k) execution time of each iteration
    * Wdist: (1 x iterations_k) evolution of the approximated Wasserstein barycenter distance
    * l_precisionB: (1 x iterations_k) evolution of the evaluation of the accuracy of the approximated Wasserstein barycenter distance
    * Precision: (1 x iterations_k) evolution of the norm max stopping criterion 
    * Iterations: (1 x iterations_k) number of the saved iterations
    * total_time: (float) total computation time
    *iterations_k: (int) number of iterations 

    NB: it is possible to save the Pi's: the transport matrices between p and the probability densities

    Infos:
    Refer to the article: "Mimouni, D., Malisani, P., Zhu, J., & de Oliveira, W. (2024). Computing Wasserstein Barycenter
    via operator splitting: the method of averaged marginals. arXiv preprint arXiv:2309.05315."

    (c) Daniel Mimouni 2024
    """""
    if exact:
        print("You are not using the correct piece of code, this code is tailor made for the reduction tree algorithm, consult https://ifpen-gitlab.appcollaboratif.fr/detocs/mam_wb.")

    st00 = time.time()

    # Parameter initializations
    # Number of probability distributions
    M = len(b)
    R = M_dist[0].shape[0]

    # storage of the transport plans
    theta = {}
    S = {}
    sum_theta_mean = np.zeros(R)
    inv_sum_S = 0
    # iterate over probabilities
    for m in range(M):
        S[m] = np.sum(b[m] > 0)
        inv_sum_S = inv_sum_S + 1 / S[m]

        # Stored in a dictionnary
        theta[m] = -1 / rho * M_dist[m]

        # to compute p
        sum_theta_mean = sum_theta_mean + np.mean(theta[m], axis=1)

    # probability:
    p = sum_theta_mean / inv_sum_S

    st1 = time.time()
    if rank == 0:
        if logs:
            print(
                f'The computation time of the method starts now, but {np.round(time.time() - st00)}s have been dedicated to initialization (matrix constructions)')
            sys.stdout.flush()

    # Keep track:
    size_saving = iterations_max // evry_it
    if keep_track:
        P = np.zeros((R, size_saving + 1))  # barycentric probability measure
        Time = np.zeros(size_saving + 1)  # time per iteration
        Precision = np.zeros(size_saving + 1)  # precision after each iteration
        Wdist = np.zeros(size_saving + 1)
        l_precisionB = np.zeros(size_saving + 1)
        Iterations = np.zeros(size_saving + 1)

    Pi = theta.copy()

    # Algorithm iterations:
    spent_time = 0
    iterations_k = 0
    it_save = 0
    while (iterations_k < iterations_min) or (
            spent_time < computation_time and iterations_k < iterations_max and evol_p > precision):  # and count_stop<10):  #  # and evol_p>10**-6) : # and evol_p>10**-16
        iterations_k = iterations_k + 1

        # print('~'*20 +  f" \n Step {k} ...")
        start = time.time()

        # Initialize for inner loops
        sum_theta_mean = np.zeros(R)
        if keep_track:
            m_theta = {}
            WD = 0

        t = 1  # if balanced Wasserstein barycenter

        # PARALLELIZATION
        # iterate over probabilities
        for m in range(M):
            # index of M_dist I use
            I = b[m] > 0

            # Get the new theta[m]
            # deltaU
            deltaU = (p - np.sum(theta[m], axis=1)) / S[m]
            deltaU = np.expand_dims(deltaU, axis=1)
            # W for the projection
            theta[m] = theta[m] - 1 / rho * M_dist[m] + 2 * t * deltaU
            # W get normalized before its projection onto the simplex
            theta[m] = theta[m] / b[m][I]
            # the transport plan is un-normalized after the projection onto the simplex
            theta[m] = projection_simplex(theta[m], z=1, axis=0) * b[m][I]

            if iterations_k >= iterations_min:
                Pi[m] = theta[m].copy()

            # this is to evaluate the algorithm advancement
            if keep_track:
                m_theta[m] = np.sum(theta[m], axis=1)
                WD += np.sum(np.multiply(theta[m], M_dist[m]))

            theta[m] = theta[m] - t * deltaU

            # mean of theta:
            sum_theta_mean = sum_theta_mean + np.mean(theta[m], axis=1)  # equivalent to: np.sum(theta[m], axis=1) /S[m]


        # Stopping criterion: using the norm max
        evol_p = np.max(np.abs(p - sum_theta_mean / inv_sum_S))

        # Compute the approximated barycenter:
        p = sum_theta_mean / inv_sum_S

        # Time management: Here is the end of 1 iteration
        end = time.time()
        iteration_time = np.round((end - start), 2)

        #### Keep Track ###
        if rank == 0 and iterations_k % evry_it != 0:
            if logs:
                print(f'{iterations_k}: computed in {iteration_time}s; (?)stopping criteria={evol_p}')
                sys.stdout.flush()
        # Save datas
        if keep_track and iterations_k % evry_it == 0:
            # show avancement of the barycenter
            if rank == 0:
                if visualize:
                    plt.close()
                    plt.figure()
                    plt.imshow(np.reshape(p, (int(R ** .5), int(R ** .5))), cmap='hot_r')
                    plt.colorbar()
                    plt.pause(0.1)

            # Compute some precision variables
            theta1m = np.zeros(R)
            for m in range(M):
                theta1m = theta1m + m_theta[m] / S[m]
            theta1m = theta1m / inv_sum_S
            # Compute the distance of the marginals of theta after the projection onto the simplex, to the subset B:
            distB = 0
            for m in range(M):
                distB += np.linalg.norm(m_theta[m] - theta1m) ** 2 / S[m]
            distB = distB ** 0.5

            # Fill the matrices
            P[:, it_save] = p
            Time[it_save] = iteration_time
            Precision[it_save] = evol_p
            Wdist[it_save] = WD
            l_precisionB[it_save] = distB
            Iterations[it_save] = iterations_k
            it_save += 1

            # Save results
            if rank == 0:
                if logs:
                    print(
                        f'{iterations_k}: computed in {iteration_time}s, WD = {WD}, with distB={distB}; (?)stopping criteria={evol_p}')
                    sys.stdout.flush()
                # save avancement:
                # save also ? P,
                l_res = [p, P, Time, Wdist, l_precisionB, Precision, Iterations, np.round((end - st1) / 60, 2),
                         iterations_k]
                with open(name, 'wb') as f:
                    pickle.dump(l_res, f)

        # manage time at a global scale:
        spent_time = end - st1

    end = time.time()
    # total_time = np.round((end - st1) / 60, 2)
    if logs:
        print(iterations_k)


    # Output
    return (p, Pi)








