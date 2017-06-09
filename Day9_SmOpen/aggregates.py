'''
------------------------------------------------------------------------
This module contains the functions that generate aggregate variables in
the steady-state or in the transition path of the overlapping
generations model with S-period lived agents and endogenous labor supply
from Chapter 7 of the OG textbook.

This Python module imports the following module(s): None

This Python module defines the following function(s):
    get_L()
    get_K()
    get_Y()
    get_C()
------------------------------------------------------------------------
'''
# Import packages
import numpy as np

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def get_L_s(narr):
    '''
    --------------------------------------------------------------------
    Solve for steady-state aggregate labor L supply or time path of aggregate
    labor supply L_t
    --------------------------------------------------------------------
    INPUTS:
    narr = (S,) vector or (S, T_S-1) matrix, values for steady-state
           labor supply (n1, n2, ...nS) or time path of the distribution
           of labor supply

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    L = scalar > 0, aggregate labor

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: L
    --------------------------------------------------------------------
    '''
    if narr.ndim == 1:  # This is the steady-state case
        L = narr.sum()
    elif narr.ndim == 2:  # This is the time path case
        L = narr.sum(axis=0)

    return L


def get_K_s(barr):
    '''
    --------------------------------------------------------------------
    Solve for steady-state aggregate capital stock supplied K or time path of
    aggregate capital stock supplied K_t.

    We have included a stitching function for K when K<=0 such that the
    the adjusted value is:

    K_adj = f(K) = eps * exp(K- eps)

    This function has the properties that
    (i) f(K)>0 for all K,
    (ii) f(eps) = eps
    (iii) f'(eps) = 1
    (iv) f'(K) > 0 for all K
    --------------------------------------------------------------------
    INPUTS:
    barr = (S,) vector or (S, T+S-1) matrix, values for steady-state
           savings (b_1, b_2,b_3,...b_S) or time path of the
           distribution of savings

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    epsilon = scalar > 0, small value at which stitch f(K) function
    K       = scalar or (T+S-2,) vector, steady-state aggregate capital
              stock or time path of aggregate capital stock
    K_cnstr = Boolean or (T+S-2) Boolean, =True if K <= 0 or if K_t <= 0

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: K, K_cnstr
    --------------------------------------------------------------------
    '''
    epsilon = 0.1
    if barr.ndim == 1:  # This is the steady-state case
        K = barr.sum()
        K_cnstr = K <= 0
        if K_cnstr:
            print('get_K() warning: distribution of savings and/or ' +
                  'parameters created K<=0')
            # Force K > 0 by stitching eps * exp(K - eps) for K <= 0
            K = epsilon * np.exp(K - epsilon)

    elif barr.ndim == 2:  # This is the time path case
        K = barr.sum(axis=0)
        K_cnstr = K <= 0
        if K.min() <= 0:
            print('Aggregate capital constraint is violated K<=0 for ' +
                  'some period in time path.')
            K[K_cnstr] = epsilon * np.exp(K[K_cnstr] - epsilon)

    return K, K_cnstr


def get_Y(params, K, L):
    '''
    --------------------------------------------------------------------
    Solve for steady-state aggregate output Y or time path of aggregate
    output Y_t
    --------------------------------------------------------------------
    INPUTS:
    params = length 2 tuple, production function parameters
             (A, alpha)
    A      = scalar > 0, total factor productivity
    alpha  = scalar in (0,1), capital share of income
    K      = scalar > 0 or (T+S-2,) vector, aggregate capital stock
             or time path of the aggregate capital stock
    L      = scalar > 0 or (T+S-2,) vector, aggregate labor or time
             path of the aggregate labor

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    Y = scalar > 0 or (T+S-2,) vector, aggregate output (GDP) or
        time path of aggregate output (GDP)

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: Y
    --------------------------------------------------------------------
    '''
    A, alpha = params
    Y = A * (K ** alpha) * (L ** (1 - alpha))

    return Y


def get_C(carr):
    '''
    --------------------------------------------------------------------
    Solve for steady-state aggregate consumption C or time path of
    aggregate consumption C_t
    --------------------------------------------------------------------
    INPUTS:
    carr = (S,) vector or (S, T) matrix, distribution of consumption c_s
           in steady state or time path for the distribution of
           consumption

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    C = scalar > 0 or (T,) vector, aggregate consumption or time path of
        aggregate consumption

    Returns: C
    --------------------------------------------------------------------
    '''
    if carr.ndim == 1:
        C = carr.sum()
    elif carr.ndim == 2:
        C = carr.sum(axis=0)

    return C
