'''
------------------------------------------------------------------------
This module contains the functions used to solve for the values of
tax functions. This module is used to solve for the model with S-period
lived agents and endogenous labor supply and an unbalanced government
budget constraint from Chapter 15 of the OG textbook.

This Python module imports the following module(s):
    aggregates.py

This Python module defines the following function(s):
    revenue()

------------------------------------------------------------------------
'''

# Packages
import numpy as np
import aggregates as aggr

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''



def revenue(r, w, b_s, n, K, L, params):
    '''
    --------------------------------------------------------------------
    Solve for total tax revenue.
    --------------------------------------------------------------------
    INPUTS:
    r      = scalar or (T2+S-1,) vector, real interest rate
    w      = scalar > 0 or (T2+S-1,) vector, real wage rate
    b_s    = (S,T2+S-1) array, distribution of household savings
    n      = (S,T2+S-1) array, distribution of household labor supply
    K      = scalar > 0 or (T2+S-1,) vector, aggregate capital stock
    L      = scalar > 0 or (T2+S-1,) vector, aggregate labor supply
    params = length 5 tuple,
                (A, alpha, delta, tax_params)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        aggr.get_Y()


    OBJECTS CREATED WITHIN FUNCTION:
    A           = scalar > 0, total factor productivity parameter in
                    firms' production function
    alpha       = scalar in (0,1), capital share of income
    delta       = scalar in [0,1], per-period capital depreciation rt
    tax_params  = length 3 tuple, (tau_l, tau_k, tau_c)
    tau_l       = scalar, marginal tax rate on labor income
    tau_k       = scalar, marginal tax rate on capital income
    tau_c       = scalar, marginal tax rate on corporate income
    Y_params    = length 2 tuple, (A, alpha)
    Y           = scalar > 0 or (T2+S-1,) vector, aggregate output
    cit_revenue = scalar >= 0 or (T2+S-1,) vector, corporate income tax revenue
    iit_revenue = scalar >= 0 or (T2+S-1,) vector, individual income tax revenue
    R           = scalar >= 0 or (T2+S-1,) vector, total tax revenue


    FILES CREATED BY THIS FUNCTION:

    RETURNS: R
    --------------------------------------------------------------------
    '''
    A, alpha, delta, tax_params = params
    tau_l, tau_k, tau_c = tax_params
    Y_params = (A, alpha)
    Y = aggr.get_Y(Y_params, K, L)
    cit_revenue = tau_c*(Y-w*L) - tau_c*delta*K
    iit_revenue = (tau_l*w*n).sum(axis=0) + (tau_k*r*b_s).sum(axis=0)

    R = cit_revenue + iit_revenue

    return R
