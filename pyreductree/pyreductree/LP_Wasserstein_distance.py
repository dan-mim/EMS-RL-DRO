# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 10:44:08 2023

@author: mimounid
"""

from scipy.sparse import csc_matrix
import scipy
from scipy import optimize
import time
import numpy as np


__all__ = ["Wasserstein_distance_LP"]

# This function compute the exact Wasserstein distance between two probability distributions
# It runs the massive linear program with the HiGHS method
def Wasserstein_distance_LP(p, q, M_dist):
    """
    Inputs:
    *p: (n x 1) probability measure
    *q: (m x 1) probability measure
    *M_dist: (n x m) matrix giving distances between the locations of supports p and q
    
    Outputs:
    *Distance_exact: (float) Wasserstein distance between p and q
    *Pi_exact: (n x m) Transport matrix between p and q
    *Time: (float) time spent to compute the result
    
    Infos:
    LP resolution solving: Wasserstein_distance(p,q) (p and q have the same support)
    *function to minimize: c @ x
    *constraints: A @ x = p and B @ x = q
    The massive linear program is run with the HiGHS method (see more on HiGHS.dev)
    """
    # Time management:
    start = time.time()
    
    
    # Dimensions
    # R_ = len(p)
    R_ = M_dist.shape[0]
    S_ = M_dist.shape[1] #len(q)

    # Distance matrix as a vector c.T (1,R_*S_)
    c = np.reshape(M_dist, (R_*S_,))

    # Building A: Constraint (right sum of Pi to get p)
    shape_A = (R_, R_*S_)
    indices_values_A = np.ones(R_*S_)
    # lines indices: time_spent = 8s for R_=S_=1600
    line_indices_A = np.array([])
    for r in range(R_):
        line_indices_A = np.append(line_indices_A, np.ones(S_) * r)
    # column indices: time_spent < 0.1s for R_=S_=1600
    column_indices_A = np.linspace(0,R_*S_-1, R_*S_)

    # Building B: Constraint (left sum of Pi to get q)
    shape_B = (S_, R_*S_)
    indices_values_B = np.ones(R_*S_)
    # lines indices: time_spent for R_=S_=1600
    line_indices_B = np.array([])
    for s in range(S_):
        line_indices_B = np.append(line_indices_B, np.ones(R_) * s)
    # column indices: time_spent for R_=S_=1600
    column_indices_B = np.array([])
    for s in range(S_):
        column_indices_B = np.append(column_indices_B, np.linspace(0,S_*(R_-1),R_) + s)

    # Building A_eq = concatenate((A, B), axis=0)
    shape_Aeq = (R_ + S_, R_ * S_)
    indices_values_Aeq = np.append(indices_values_A, indices_values_B)
    line_indices_Aeq = np.append(line_indices_A, line_indices_B + R_)
    column_indices_Aeq = np.append(column_indices_A, column_indices_B)
    A_eq = csc_matrix((indices_values_Aeq, (line_indices_Aeq, column_indices_Aeq)), shape_Aeq)

    # b_eq:
    b_eq = np.append(p,q)
    
    # Resolution:
    res = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs')
   
    # computing time:
    end = time.time()
    Time = np.round((end-start), 2)
        
    # Output:
    try:
        # Optimization terminated successfully.
        Distance_exact = res.fun
        Pi_exact = res.x
        Pi_exact = np.reshape(Pi_exact, (R_, S_))
        
        return(Distance_exact, Pi_exact, Time)
   
    except:
        # The problem is infeasible.
        message = res.message
        return(message, [], Time) 


