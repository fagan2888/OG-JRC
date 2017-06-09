'''
------------------------------------------------------------------------
This module contains the functions that generate the variables
associated with firms' optimization in the steady-state or in the
transition path of the overlapping generations model with S-period lived
agents and endogenous labor supply from Chapter 7 of the OG textbook.

This Python module imports the following module(s): None

This Python module defines the following function(s):
    get_w()
    get_r()
------------------------------------------------------------------------
'''
# Import packages

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def get_w(r, params):
    '''
    --------------------------------------------------------------------
    Solve for steady-state wage w or time path of wages w_t
    --------------------------------------------------------------------
    INPUTS:
    r      = scalar, real interest rate
    params = length 2 tuple, (A, alpha)
    A      = scalar > 0, total factor productivity
    alpha  = scalar in (0, 1), capital share of income
    delta  = scalar in (0, 1), per period depreciation rate

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    w = scalar > 0 or (T+S-2) vector, steady-state wage or time path of
        wage

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: w
    --------------------------------------------------------------------
    '''
    A, alpha, delta = params
    w = ((1 - alpha) * A * ((alpha * A) / (r + delta)) ** (alpha / (1 - alpha)))

    return w


def get_r(params, K, L):
    '''
    --------------------------------------------------------------------
    Solve for steady-state interest rate r or time path of interest
    rates r_t
    --------------------------------------------------------------------
    INPUTS:
    params = length 3 tuple, (A, alpha, delta)
    A      = scalar > 0, total factor productivity
    alpha  = scalar in (0, 1), capital share of income
    delta  = scalar in (0, 1), per period depreciation rate
    K      = scalar > 0 or (T+S-2,) vector, steady-state aggregate
             capital stock or time path of the aggregate capital stock
    L      = scalar > 0 or (T+S-2,) vector, steady-state aggregate
             labor or time path of aggregate labor

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    r = scalar > 0 or (T+S-2) vector, steady-state interest rate or time
        path of interest rate

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: r
    --------------------------------------------------------------------
    '''
    A, alpha, delta = params
    r = alpha * A * ((L / K) ** (1 - alpha)) - delta

    return r

def get_K_d(r, L, params):
    '''
    --------------------------------------------------------------------
    Solve for steady-state capital demand K_d or time path of capital
    demand K_d_t
    --------------------------------------------------------------------
    INPUTS:
    params = length 3 tuple, (A, alpha, delta)
    A      = scalar > 0, total factor productivity
    alpha  = scalar in (0, 1), capital share of income
    delta  = scalar in (0, 1), per period depreciation rate
    L      = scalar > 0 or (T+S-2,) vector, steady-state aggregate
             labor or time path of aggregate labor

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    K_d = scalar > 0 or (T+S-2) vector, steady-state capital demand or time
        path of capital demand

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: K_d
    --------------------------------------------------------------------
    '''

    A, alpha, delta = params
    K_d = ((alpha * A) / (r + delta)) ** (1 / (1 - alpha)) * L

    return K_d
