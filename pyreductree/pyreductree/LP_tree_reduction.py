
# RECURSIVE COMPUTATION : SOLVE THE LINEAR PROGRAM
import numpy as np
from scipy.sparse import csc_matrix, kron, hstack, vstack, eye
import scipy
from scipy import optimize
import time


__all__ = ["LP_reduction_nt"]


def LP_reduction_nt(c, b):
    """
    This function is a linear program using the method High to compute (17) from the paper
    'Tree approximation for discrete time stochastic process: a process distance approach' from Kovacevic and Pichler
    Its aim is to recursively compute the distance between two trees.
    :param c: Distance matrix between i_s and j_s (R x S)
    :param b: list of constraints b[m=m]=(P(i|m))i>0 for every i b = [b[m=1], b[m=2],...]
    :return:
    Pi_exact: Conditional transport matrices, stacked with regard to m: (Pi_m1[j,i|n,m1] | Pi_m2 | ...)
    S: list of the number of children for each node m
    Time: spent time to compute the Linear Program
    """
    # Time management:
    start = time.time()

    # dimension parameters
    M = len(b)
    S = [len(bi) for bi in b]
    sumS = np.sum(S)

    # Objective function
    R = c.shape[0]
    c = np.reshape(c, (R * sumS,))

    # Building A: Constraint (right sum of Pi to get p)
    A = eye(sumS)
    A = hstack([A] * R)

    # IF WE COMPUTE A BARYCENTER = IF THERE IS SEVERAL PARENTS = IF M != 0
    if len(S) > 1:
        # Building B: Constraint (left sum of Pi are equal)
        # lines indices:
        shape_B = ((M - 1), sumS)
        # indices values
        indices_values_B = np.append(np.ones(S[0]), - np.ones(S[1]))
        # lines indices:
        line_indices_B = np.linspace(0, S[0] + S[1] - 1, S[0] + S[1])
        # column indices:
        column_indices_B = np.zeros(S[0] + S[1])
        index = 0
        for m in range(1, M - 1):
            indices_values_B = np.append(indices_values_B, np.append(np.ones(S[m]), - np.ones(S[m + 1])))
            index = index + S[m - 1]
            line_indices_B = np.append(line_indices_B, np.linspace(0, S[m] + S[m + 1] - 1, S[m] + S[m + 1]) + index)
            column_indices_B = np.append(column_indices_B, np.ones(S[m] + S[m + 1]) * m)
        # Building one small matrix
        B = csc_matrix((indices_values_B, (column_indices_B, line_indices_B)), shape_B)

        # Using Kronecker product to get the total matrix
        B = kron(eye(R), B)
        # Building A_eq = concatenate((A, B), axis=0)
        A_eq = vstack([A, B])
        # b_eq:
        b = np.concatenate(b)
        b_eq = np.append(b, np.zeros(R * (M - 1)))

    # IF WE COMPUTE (21) FOR THE ROOT = m == 0, necessary n==0 too.
    if len(S) == 1:
        # the sum of the Pi[j,i|0,0] for all i and all j = 1
        # Indeed, note that the constraint 2 of (21) cannot be applied because there is only one parent (list_m=[0]), but we know
        # that sum_j ( sum_i Pi[j,i|0,0] ) = sum_j P'(j|0) = sum_j P'(j) = 1
        B = np.ones((1,R * sumS))
        # Building A_eq = concatenate((A, B), axis=0)
        A_eq = vstack([A, B])
        # b_eq:
        b_eq = np.append(b, 1)

    # Resolution: with the bound x >= 0
    res = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0,None), method='highs')  # bounds=(0,None): the Pi cannot be negatif

    # computing time:
    end = time.time()
    Time = np.round((end - start), 2)

    # Output
    # Transport matrix:
    Pi_exact = res.x
    Pi_exact = np.reshape(Pi_exact, (R, sumS))

    return(Pi_exact, S, res.fun, Time)
