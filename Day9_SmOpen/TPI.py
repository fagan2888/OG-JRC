'''
------------------------------------------------------------------------
This module contains the functions used to solve the transition path
equilibrium using time path iteration (TPI) for the model with S-period
lived agents and endogenous labor supply from Chapter 7 of the OG
textbook.

This Python module imports the following module(s):
    aggregates.py
    firms.py
    households.py
    utilities.py

This Python module defines the following function(s):
    get_path()
    get_cnbpath()
    get_TPI()
------------------------------------------------------------------------
'''
# Import Packages
import time
import numpy as np
import aggregates as aggr
import firms
import households as hh
import utilities as utils
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def get_path(x1, xT, T, spec):
    '''
    --------------------------------------------------------------------
    This function generates a path from point x1 to point xT such that
    that the path x is a linear or quadratic function of time t.

        linear:    x = d*t + e
        quadratic: x = a*t^2 + b*t + c

    The identifying assumptions for quadratic are the following:

        (1) x1 is the value at time t=0: x1 = c
        (2) xT is the value at time t=T-1: xT = a*(T-1)^2 + b*(T-1) + c
        (3) the slope of the path at t=T-1 is 0: 0 = 2*a*(T-1) + b
    --------------------------------------------------------------------
    INPUTS:
    x1 = scalar, initial value of the function x(t) at t=0
    xT = scalar, value of the function x(t) at t=T-1
    T  = integer >= 3, number of periods of the path
    spec = string, "linear" or "quadratic"

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    cc    = scalar, constant coefficient in quadratic function
    bb    = scalar, coefficient on t in quadratic function
    aa    = scalar, coefficient on t^2 in quadratic function
    xpath = (T,) vector, parabolic xpath from x1 to xT

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: xpath
    --------------------------------------------------------------------
    '''
    if spec == "linear":
        xpath = np.linspace(x1, xT, T)
    elif spec == "quadratic":
        cc = x1
        bb = 2 * (xT - x1) / (T - 1)
        aa = (x1 - xT) / ((T - 1) ** 2)
        xpath = (aa * (np.arange(0, T) ** 2) + (bb * np.arange(0, T)) +
                 cc)

    return xpath


def get_cnbpath(params, rpath, wpath):
    '''
    --------------------------------------------------------------------
    Given time paths for interest rates and wages, this function
    generates matrices for the time path of the distribution of
    individual consumption, labor supply, savings, the corresponding
    Euler errors for the labor supply decision and the savings decision,
    and the residual error of end-of-life savings associated with
    solving each lifetime decision.
    --------------------------------------------------------------------
    INPUTS:
    params  = length 11 tuple, (S, T2, beta, sigma, l_tilde, b_ellip,
              upsilon, chi_n_vec, bvec1, TPI_tol, diff)
    rpath   = (T2+S-1,) vector, equilibrium time path of interest rate
    wpath   = (T2+S-1,) vector, equilibrium time path of the real wage

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        hh.c1_bSp1err()
        hh.get_cnb_vecs()
        hh.get_n_errors()
        hh.get_b_errors()

    OBJECTS CREATED WITHIN FUNCTION:
    S             = integer in [3,80], number of periods an individual
                    lives
    T2            = integer > S, number of periods until steady state
    beta          = scalar in (0,1), discount factor
    sigma         = scalar > 0, coefficient of relative risk aversion
    l_tilde       = scalar > 0, time endowment for each agent each
                    period
    b_ellip       = scalar > 0, fitted value of b for elliptical
                    disutility of labor
    upsilon       = scalar > 1, fitted value of upsilon for elliptical
                    disutility of labor
    chi_n_vec     = (S,) vector, values for chi^n_s
    bvec1         = (S,) vector, initial period savings distribution
    TPI_tol       = scalar > 0, tolerance level for fsolve's in TPI
    diff          = boolean, =True if want difference version of Euler
                    errors beta*(1+r)*u'(c2) - u'(c1), =False if want
                    ratio version [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
    cpath         = (S, T2+S-1) matrix, time path of the distribution of
                    consumption
    npath         = (S, T2+S-1) matrix, time path of the distribution of
                    labor supply
    bpath         = (S, T2+S-1) matrix, time path of the distribution of
                    savings
    n_err_path    = (S, T2+S-1) matrix, time path of distribution of
                    labor supply Euler errors
    b_err_path    = (S, T2+S-1) matrix, time path of distribution of
                    savings Euler errors
    bSp1_err_path = (S, T2) matrix, residual last period savings, which
                    should be close to zero in equilibrium. Nonzero
                    elements of matrix should only be in first column
                    and first row
    c1_options    = length 1 dict, options for
                    opt.root(hh.c1_bSp1err,...)
    b_err_params  = length 2 tuple, args to pass into
                    hh.get_b_errors()
    p             = integer in [1, S-1], index representing number of
                    periods remaining in a lifetime, used to solve
                    incomplete lifetimes
    c1_init       = scalar > 0, guess for initial period consumption
    c1_args       = length 10 tuple, args to pass into
                    opt.root(hh.c1_bSp1err,...)
    results_c1    = results object, solution from
                    opt.root(hh.c1_bSp1err,...)
    c1            = scalar > 0, optimal initial consumption
    cnb_args      = length 8 tuple, args to pass into
                    hh.get_cnb_vecs()
    cvec          = (p,) vector, individual lifetime consumption
                    decisions
    nvec          = (p,) vector, individual lifetime labor supply
                    decisions
    bvec          = (p,) vector, individual lifetime savings decisions
    b_Sp1         = scalar, savings in last period for next period.
                    Should be zero in equilibrium
    DiagMaskc     = (p, p) boolean identity matrix
    DiagMaskb     = (p-1, p-1) boolean identity matrix
    n_err_params  = length 5 tuple, args to pass into hh.get_n_errors()
    n_err_vec     = (p,) vector, individual lifetime labor supply Euler
                    errors
    b_err_vec     = (p-1,) vector, individual lifetime savings Euler
                    errors
    t             = integer in [0,T2-1], index of time period (minus 1)

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: cpath, npath, bpath, n_err_path, b_err_path, bSp1_err_path
    --------------------------------------------------------------------
    '''
    (S, T2, beta, sigma, l_tilde, b_ellip, upsilon, chi_n_vec,
        bvec1, TPI_tol, diff) = params
    cpath = np.zeros((S, T2 + S - 1))
    npath = np.zeros((S, T2 + S - 1))
    bpath = np.append(bvec1.reshape((S, 1)),
                      np.zeros((S, T2 + S - 2)), axis=1)
    n_err_path = np.zeros((S, T2 + S - 1))
    b_err_path = np.zeros((S, T2 + S - 1))
    bSp1_err_path = np.zeros((S, T2))
    # Solve the incomplete remaining lifetime decisions of agents alive
    # in period t=1 but not born in period t=1
    c1_options = {'maxiter': 500}
    b_err_params = (beta, sigma)
    for p in range(1, S):
        c1_init = 0.1
        c1_args = (bvec1[-p], beta, sigma, l_tilde, b_ellip, upsilon,
                   chi_n_vec[-p:], rpath[:p], wpath[:p], diff)
        results_c1 = \
            opt.root(hh.c1_bSp1err, c1_init, args=(c1_args),
                     method='lm', tol=TPI_tol, options=(c1_options))
        c_1 = results_c1.x
        cnb_args = (bvec1[-p], beta, sigma, l_tilde, b_ellip, upsilon,
                    chi_n_vec[-p:], diff)
        cvec, nvec, bvec, b_Sp1 = \
            hh.get_cnb_vecs(c_1, rpath[:p], wpath[:p], cnb_args)
        DiagMaskc = np.eye(p, dtype=bool)
        DiagMaskb = np.eye(p - 1, dtype=bool)
        cpath[-p:, :p] = DiagMaskc * cvec + cpath[-p:, :p]
        npath[-p:, :p] = DiagMaskc * nvec + npath[-p:, :p]
        n_err_args = (wpath[:p], cvec, sigma, l_tilde, chi_n_vec[-p:], b_ellip,
                      upsilon, diff)
        n_err_vec = hh.get_n_errors(nvec, n_err_args)
        n_err_path[-p:, :p] = (DiagMaskc * n_err_vec +
                               n_err_path[-p:, :p])
        bSp1_err_path[-p, 0] = b_Sp1
        if p > 1:
            bpath[S - p + 1:, 1:p] = (DiagMaskb * bvec[1:] +
                                      bpath[S - p + 1:, 1:p])
            b_err_vec = hh.get_b_errors(b_err_params, rpath[1:p], cvec,
                                        diff)
            b_err_path[S - p + 1:, 1:p] = (DiagMaskb * b_err_vec +
                                           b_err_path[S - p + 1:, 1:p])
            # print('p=', p, ', max. abs. all errs: ',
            #       np.hstack((n_err_vec, b_err_vec, b_Sp1)).max())

    # Solve the remaining lifetime decisions of agents born between
    # period t=1 and t=T (complete lifetimes)
    DiagMaskc = np.eye(S, dtype=bool)
    DiagMaskb = np.eye(S - 1, dtype=bool)
    for t in range(T2):  # Go from periods 1 to T (columns 0 to T-1)
        if t == 0:
            c1_init = 0.1
        else:
            c1_init = cpath[0, t - 1]
        c1_args = (0.0, beta, sigma, l_tilde, b_ellip, upsilon,
                   chi_n_vec, rpath[t:t + S], wpath[t:t + S], diff)
        results_c1 = \
            opt.root(hh.c1_bSp1err, c1_init, args=(c1_args),
                     method='lm', tol=TPI_tol, options=(c1_options))
        c_1 = results_c1.x
        cnb_args = (0.0, beta, sigma, l_tilde, b_ellip, upsilon,
                    chi_n_vec, diff)
        cvec, nvec, bvec, b_Sp1 = \
            hh.get_cnb_vecs(c_1, rpath[t:t + S], wpath[t:t + S],
                            cnb_args)
        cpath[:, t:t + S] = DiagMaskc * cvec + cpath[:, t:t + S]
        npath[:, t:t + S] = DiagMaskc * nvec + npath[:, t:t + S]
        n_err_args = (wpath[t:t + S], cvec, sigma, l_tilde, chi_n_vec, b_ellip, upsilon, diff)
        n_err_vec = hh.get_n_errors(nvec, n_err_args)
        n_err_path[:, t:t + S] = (DiagMaskc * n_err_vec +
                                  n_err_path[:, t:t + S])
        bpath[:, t:t + S] = DiagMaskc * bvec + bpath[:, t:t + S]
        b_err_vec = hh.get_b_errors(b_err_params, rpath[t + 1:t + S],
                                    cvec, diff)
        b_err_path[1:, t + 1:t + S] = (DiagMaskb * b_err_vec +
                                       b_err_path[1:, t + 1:t + S])
        bSp1_err_path[0, t] = b_Sp1
        # print('t=', t, ', max. abs. all errs: ',
        #       np.absolute(np.hstack((n_err_vec, b_err_vec,
        #                              b_Sp1))).max())

    return cpath, npath, bpath, n_err_path, b_err_path, bSp1_err_path


def firstdoughnutring(guesses, args):
    '''
    Solves the first entries of the upper triangle of the twist doughnut.  This is
    separate from the main TPI function because the the values of b and n are scalars,
    so it is easier to just have a separate function for these cases.
    Inputs:
        guesses = guess for b and n (2x1 list)
        winit = initial wage rate (scalar)
        rinit = initial rental rate (scalar)
        BQinit = initial aggregate bequest (scalar)
        T_H_init = initial lump sum tax (scalar)
        initial_b = initial distribution of capital (SxJ array)
        factor = steady state scaling factor (scalar)
        j = which ability type is being solved for (scalar)
        parameters = list of parameters (list)
        theta = replacement rates (Jx1 array)
        tau_bq = bequest tax rates (Jx1 array)
    Output:
        euler errors (2x1 list)
    '''

    # unpack tuples of parameters
    r, w, S, T2, beta, sigma, l_tilde, b_ellip, upsilon, chi_n_vec,\
        initial_b, diff = args


    b_splus1 = 0.0 # leave zero savings in last period of life
    n = float(guesses[1])
    b_s = float(initial_b[-1])

    # Euler equations
    cons = hh.get_cons(r, w, b_s, b_splus1, n)
    b_params = (beta, sigma)
    error1 = 0.0 #hh.get_b_errors(b_params, r, cons, diff)

    n_args = (w, cons, sigma, l_tilde, chi_n_vec, b_ellip, upsilon, diff)
    error2 = hh.get_n_errors(n, n_args)

    if n <= 0 or n >= 1:
        error2 += 1e14
    if cons <= 0:
        error1 += 1e14
    return [error1] + [error2]


def twist_doughnut(guesses, td_args):
    '''
    Parameters:
        guesses = distribution of capital and labor (various length list)
        w   = wage rate ((T+S)x1 array)
        r   = rental rate ((T+S)x1 array)
        BQ = aggregate bequests ((T+S)x1 array)
        T_H = lump sum tax over time ((T+S)x1 array)
        factor = scaling factor (scalar)
        j = which ability type is being solved for (scalar)
        s = which upper triangle loop is being solved for (scalar)
        t = which diagonal is being solved for (scalar)
        params = list of parameters (list)
        theta = replacement rates (Jx1 array)
        tau_bq = bequest tax rate (Jx1 array)
        rho = mortalit rate (Sx1 array)
        lambdas = ability weights (Jx1 array)
        e = ability type (SxJ array)
        initial_b = capital stock distribution in period 0 (SxJ array)
        chi_b = chi^b_j (Jx1 array)
        chi_n = chi^n_s (Sx1 array)
    Output:
        Value of Euler error (various length list)
    '''
    rpath, wpath, s, t, S, T2, beta, sigma, l_tilde, b_ellip, upsilon, \
                chi_n_vec, initial_b, diff = td_args

    length = len(guesses) // 2
    b_guess = np.array(guesses[:length])
    n_guess = np.array(guesses[length:])

    b_guess[-1] = 0.0 # save nothing in last period

    if length == S:
        b_s = np.array([0] + list(b_guess[:-1]))
    else:
        # b_s = np.array([(initial_b[-(s + 3)])] + list(b_guess[:-1]))
        b_s = np.array([(initial_b[-(s + 2)])] + list(b_guess[:-1]))

    w = wpath[t:t + length]
    r = rpath[t:t + length]


    # Euler equations
    cons = hh.get_cons(r, w, b_s, b_guess, n_guess)

    b_params = (beta, sigma)
    euler_errors = hh.get_b_errors(b_params, r[1:], cons, diff)
    error1 = np.append(euler_errors,0.0)

    n_args = (w, cons, sigma, l_tilde, chi_n_vec[-length:], b_ellip, upsilon, diff)
    error2 = hh.get_n_errors(n_guess, n_args)

    # Check and punish constraint violations
    mask1 = n_guess < 0
    error2[mask1] += 1e14
    mask2 = n_guess > l_tilde
    error2[mask2] += 1e14
    mask3 = cons < 0
    error2[mask3] += 1e14

    return list(error1.flatten()) + list(error2.flatten())


def solve_bn_path(guesses, args):
    '''
    --------------------------------------------------------------------
    Solves for transition path equilibrium using time path iteration
    (TPI)
    --------------------------------------------------------------------
    INPUTS:
    params = length 21 tuple, (S, T1, T2, beta, sigma, l_tilde, b_ellip,
             upsilon, chi_n_vec, A, alpha, delta, K_ss, L_ss, C_ss,
             maxiter, mindist, TPI_tol, xi, diff, hh_fsolve)
    bvec1  = (S,) vector, initial period savings distribution
    graphs = Boolean, =True if want graphs of TPI objects

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        aggr.get_K()
        get_path()
        firms.get_r()
        firms.get_w()
        get_cnbpath()
        bn_solve
        aggr.get_L()
        aggr.get_Y()
        aggr.get_C()
        utils.print_time()

    OBJECTS CREATED WITHIN FUNCTION:
    start_time    = scalar, current processor time in seconds (float)
    S             = integer in [3,80], number of periods an individual
                    lives
    T1            = integer > S, number of time periods until steady
                    state is assumed to be reached
    T2            = integer > T1, number of time periods after which
                    steady-state is forced in TPI
    beta          = scalar in (0,1), discount factor for model period
    sigma         = scalar > 0, coefficient of relative risk aversion
    l_tilde       = scalar > 0, time endowment for each agent each
                    period
    b_ellip       = scalar > 0, fitted value of b for elliptical
                    disutility of labor
    upsilon       = scalar > 1, fitted value of upsilon for elliptical
                    disutility of labor
    chi_n_vec     = (S,) vector, values for chi^n_s
    A             = scalar > 0, total factor productivity parameter in
                    firms' production function
    alpha         = scalar in (0,1), capital share of income
    delta         = scalar in [0,1], per-period capital depreciation rt
    K_ss          = scalar > 0, steady-state aggregate capital stock
    L_ss          = scalar > 0, steady-state aggregate labor supply
    C_ss          = scalar > 0, steady-state aggregate consumption
    maxiter       = integer >= 1, Maximum number of iterations for TPI
    mindist       = scalar > 0, convergence criterion for TPI
    TPI_tol       = scalar > 0, tolerance level for TPI root finders
    xi            = scalar in (0,1], TPI path updating parameter
    diff          = Boolean, =True if want difference version of Euler
                    errors beta*(1+r)*u'(c2) - u'(c1), =False if want
                    ratio version [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
    hh_fsolve     = boolean, =True if solve inner-loop household problem by
                    choosing c_1 to set final period savings b_{S+1}=0.
                    Otherwise, solve the household problem as multivariate
                    root finder with 2S-1 unknowns and equations
    K1            = scalar > 0, initial aggregate capital stock
    K1_cnstr      = Boolean, =True if K1 <= 0
    Kpath_init    = (T2+S-1,) vector, initial guess for the time path of
                    the aggregate capital stock
    Lpath_init    = (T2+S-1,) vector, initial guess for the time path of
                    aggregate labor
    domain        = (T2,) vector, integers from 0 to T2-1
    domain2       = (T2,S) array, integers from 0 to T2-1 repeated S times
    ending_b      = (S,) vector, distribution of savings at end of time path
    initial_b     = (S,) vector, distribution of savings in initial period
    guesses_b     = (T2,S) array, initial guess at distribution of savings
                    over the time path
    ending_b_tail = (S,S) array, distribution of savings for S periods after
                     end of time path
    guesses_b     = (T2+S,S) array, guess at distribution of savings
                    for T2+S periods
    domain3       = (T2,S) array, integers from 0 to T2-1 repeated S times
    initial_n     = (S,) vector, distribution of labor supply in initial period
    guesses_n     = (T2,S) array, initial guess at distribution of labor
                    supply over the time path
    ending_n_tail = (S,S) array, distribution of labor supply for S periods after
                     end of time path
    guesses_n     = (T2+S,S) array, guess at distribution of labor supply
                    for T2+S periods
    guesses       = length 2 tuple, initial guesses at distributions of savings
                    and labor supply over the time path
    iter_TPI      = integer >= 0, current iteration of TPI
    dist          = scalar >= 0, distance measure between initial and
                    new paths
    r_params      = length 3 tuple, (A, alpha, delta)
    w_params      = length 2 tuple, (A, alpha)
    Y_params      = length 2 tuple, (A, alpha)
    cnb_params    = length 11 tuple, args to pass into get_cnbpath()
    rpath         = (T2+S-1,) vector, time path of the interest rates
    wpath         = (T2+S-1,) vector, time path of the wages
    ind           = (S,) vector, integers from 0 to S
    bn_args       = length 14 tuple, arguments to be passed to bn_solve()
    cpath         = (S, T2+S-1) matrix, time path of distribution of
                    individual consumption c_{s,t}
    npath         = (S, T2+S-1) matrix, time path of distribution of
                    individual labor supply n_{s,t}
    bpath         = (S, T2+S-1) matrix, time path of distribution of
                    individual savings b_{s,t}
    n_err_path    = (S, T2+S-1) matrix, time path of distribution of
                    individual labor supply Euler errors
    b_err_path    = (S, T2+S-1) matrix, time path of distribution of
                    individual savings Euler errors. First column and
                    first row are identically zero
    bSp1_err_path = (S, T2) matrix, residual last period savings, which
                    should be close to zero in equilibrium. Nonzero
                    elements of matrix should only be in first column
                    and first row
    Kpath_new     = (T2+S-1,) vector, new path of the aggregate capital
                    stock implied by household and firm optimization
    Kpath_cnstr   = (T2+S-1,) Boolean vector, =True if K_t<=0
    Lpath_new     = (T2+S-1,) vector, new path of the aggregate labor
    rpath_new     = (T2+S-1,) vector, updated time path of interest rate
    wpath_new     = (T2+S-1,) vector, updated time path of the wages
    Ypath         = (T2+S-1,) vector, equilibrium time path of aggregate
                    output (GDP) Y_t
    Cpath         = (T2+S-1,) vector, equilibrium time path of aggregate
                    consumption C_t
    RCerrPath     = (T2+S-2,) vector, equilibrium time path of the
                    resource constraint error:
                    Y_t - C_t - K_{t+1} + (1-delta)*K_t
    KL_path_new   = (2*T2,) vector, appended K_path_new and L_path_new
                    from observation 1 to T2
    KL_path_init  = (2*T2,) vector, appended K_path_init and L_path_init
                    from observation 1 to T2
    Kpath         = (T2+S-1,) vector, equilibrium time path of aggregate
                    capital stock K_t
    Lpath         = (T2+S-1,) vector, equilibrium time path of aggregate
                    labor L_t
    tpi_time      = scalar, time to compute TPI solution (seconds)
    tpi_output    = length 14 dictionary, {cpath, npath, bpath, wpath,
                    rpath, Kpath, Lpath, Ypath, Cpath, bSp1_err_path,
                    n_err_path, b_err_path, RCerrPath, tpi_time}

    FILES CREATED BY THIS FUNCTION:
        Kpath.png
        Lpath.png
        Ypath.png
        C_aggr_path.png
        wpath.png
        rpath.png
        cpath.png
        npath.png
        bpath.png

    RETURNS: tpi_output
    --------------------------------------------------------------------
    '''


    guesses_b, guesses_n = guesses

    # initialize arrays
    (rpath, wpath, ind, S, T2, beta, sigma, l_tilde, b_ellip, upsilon, chi_n_vec,
        initial_b, TPI_tol, diff) = args
    cpath = np.zeros((T2 + S - 1, S))
    npath = np.zeros((T2 + S - 1, S))
    bpath = np.zeros((T2 + S - 1, S))
    # bpath = np.append(initial_b.reshape((1, S)),
    #                   np.zeros((T2 + S - 2, S)), axis=1)
    n_err_path = np.zeros((T2 + S - 1, S))
    b_err_path = np.zeros((T2 + S - 1, S))

    # first doughnut ring
    first_doughnut_args = [rpath[0], wpath[0], S, T2, beta, sigma, l_tilde, b_ellip, upsilon, \
        chi_n_vec[-1], initial_b, diff]
    [solutions, infodict, ier, message] =\
                                        opt.fsolve(firstdoughnutring,
                                        [guesses_b[0, -1], guesses_n[0, -1]],
                                        args=(first_doughnut_args),
                                        xtol=TPI_tol, full_output=True)
    bpath[0, -1], npath[0, -1] = solutions
    b_err_path[0, -1], n_err_path[0, -1] = infodict['fvec']
    #print('Upper triangle: ', b_err_path[0, -1], n_err_path[0, -1])


    for s in range(S - 2):  # Upper triangle
        ind2 = np.arange(s + 2)
        b_guesses_to_use = np.diag(
            guesses_b[:S, :], S - (s + 2))
        n_guesses_to_use = np.diag(guesses_n[:S, :], S - (s + 2))

        # initialize array of diagonal elements
        td1_args = [rpath, wpath, s, 0, S, T2, beta, sigma, l_tilde,
                      b_ellip, upsilon, chi_n_vec, initial_b, diff]
        [solutions, infodict, ier, message] = opt.fsolve(twist_doughnut, list(
            b_guesses_to_use) + list(n_guesses_to_use), args=(td1_args),
            xtol=TPI_tol, full_output=True)

        bpath[ind2, S - (s + 2) + ind2] = solutions[:len(solutions) // 2]
        bpath[ind2, -1] = 0.0
        npath[ind2, S - (s + 2) + ind2] = solutions[len(solutions) // 2:]
        b_err_path[ind2, S - (s + 2) + ind2] = infodict['fvec'][:len(solutions) // 2]
        b_err_path[ind2, -1] = 0.0
        n_err_path[ind2, S - (s + 2) + ind2] = infodict['fvec'][len(solutions) // 2:]

        # print('First twist: ', b_err_path[ind2, S - (s + 2) + ind2], n_err_path[ind2, S - (s + 2) + ind2])

    for t in range(0, T2):
        b_guesses_to_use = .75 * \
            np.diag(guesses_b[t:t + S, :])
        n_guesses_to_use = np.diag(guesses_n[t:t + S, :])

        # initialize array of diagonal elements
        td2_args = [rpath, wpath, None, t, S, T2, beta, sigma, l_tilde,
                      b_ellip, upsilon, chi_n_vec, initial_b, diff]
        [solutions, infodict, ier, message] = opt.fsolve(twist_doughnut, list(
            b_guesses_to_use) + list(n_guesses_to_use), args=(td2_args),
            xtol=TPI_tol, full_output=True)

        bpath[t + ind, ind] = solutions[:S]
        bpath[t + ind, -1] = 0.0
        npath[t + ind, ind] = solutions[S:]
        b_err_path[t + ind, ind] = infodict['fvec'][:S]
        b_err_path[t + ind, -1] = 0.0
        n_err_path[t + ind, ind] = infodict['fvec'][S:]
        # print('Second twist: ', b_err_path[t + ind, ind], n_err_path[t + ind, ind] )
        # if (np.absolute(b_err_path).max() > 1e-7) or (np.absolute(b_err_path).max() > 1e-7):
        #     err_msg = ('ERROR: error too large')
        #     print(np.max(ind), t, bpath[t + ind, ind].shape)
        #     raise RuntimeError(err_msg)

    # print('Max errors: ', np.absolute(n_err_path).max(), np.absolute(b_err_path).max())

    return  np.transpose(npath), np.transpose(bpath), np.transpose(n_err_path), np.transpose(b_err_path)


def get_TPI(params, bvec1, graphs):
    '''
    --------------------------------------------------------------------
    Solves for transition path equilibrium using time path iteration
    (TPI)
    --------------------------------------------------------------------
    INPUTS:
    params = length 20 tuple, (S, T1, T2, beta, sigma, l_tilde, b_ellip,
             upsilon, chi_n_vec, A, alpha, delta, K_ss, L_ss, C_ss,
             maxiter, mindist, TPI_tol, xi, diff)
    bvec1  = (S,) vector, initial period savings distribution
    graphs = Boolean, =True if want graphs of TPI objects

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        aggr.get_K()
        get_path()
        firms.get_r()
        firms.get_w()
        get_cnbpath()
        aggr.get_L()
        aggr.get_Y()
        aggr.get_C()
        utils.print_time()

    OBJECTS CREATED WITHIN FUNCTION:
    start_time    = scalar, current processor time in seconds (float)
    S             = integer in [3,80], number of periods an individual
                    lives
    T1            = integer > S, number of time periods until steady
                    state is assumed to be reached
    T2            = integer > T1, number of time periods after which
                    steady-state is forced in TPI
    beta          = scalar in (0,1), discount factor for model period
    sigma         = scalar > 0, coefficient of relative risk aversion
    l_tilde       = scalar > 0, time endowment for each agent each
                    period
    b_ellip       = scalar > 0, fitted value of b for elliptical
                    disutility of labor
    upsilon       = scalar > 1, fitted value of upsilon for elliptical
                    disutility of labor
    chi_n_vec     = (S,) vector, values for chi^n_s
    A             = scalar > 0, total factor productivity parameter in
                    firms' production function
    alpha         = scalar in (0,1), capital share of income
    delta         = scalar in [0,1], per-period capital depreciation rt
    r_star        = scalar, model-period world interest real interest rate
    K_s_ss        = scalar > 0, steady-state aggregate capital stock supplied
    L_ss          = scalar > 0, steady-state aggregate labor supply
    C_ss          = scalar > 0, steady-state aggregate consumption
    maxiter       = integer >= 1, Maximum number of iterations for TPI
    mindist       = scalar > 0, convergence criterion for TPI
    TPI_tol       = scalar > 0, tolerance level for TPI root finders
    xi            = scalar in (0,1], TPI path updating parameter
    diff          = Boolean, =True if want difference version of Euler
                    errors beta*(1+r)*u'(c2) - u'(c1), =False if want
                    ratio version [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
    K1            = scalar > 0, initial aggregate capital stock
    K1_cnstr      = Boolean, =True if K1 <= 0
    Kpath_init    = (T2+S-1,) vector, initial guess for the time path of
                    the aggregate capital stock
    Lpath_init    = (T2+S-1,) vector, initial guess for the time path of
                    aggregate labor
    iter_TPI      = integer >= 0, current iteration of TPI
    dist          = scalar >= 0, distance measure between initial and
                    new paths
    r_params      = length 3 tuple, (A, alpha, delta)
    w_params      = length 2 tuple, (A, alpha)
    Y_params      = length 2 tuple, (A, alpha)
    cnb_params    = length 11 tuple, args to pass into get_cnbpath()
    rpath         = (T2+S-1,) vector, time path of the interest rates
    wpath         = (T2+S-1,) vector, time path of the wages
    cpath         = (S, T2+S-1) matrix, time path of distribution of
                    individual consumption c_{s,t}
    npath         = (S, T2+S-1) matrix, time path of distribution of
                    individual labor supply n_{s,t}
    bpath         = (S, T2+S-1) matrix, time path of distribution of
                    individual savings b_{s,t}
    n_err_path    = (S, T2+S-1) matrix, time path of distribution of
                    individual labor supply Euler errors
    b_err_path    = (S, T2+S-1) matrix, time path of distribution of
                    individual savings Euler errors. First column and
                    first row are identically zero
    bSp1_err_path = (S, T2) matrix, residual last period savings, which
                    should be close to zero in equilibrium. Nonzero
                    elements of matrix should only be in first column
                    and first row
    Kpath_new     = (T2+S-1,) vector, new path of the aggregate capital
                    stock implied by household and firm optimization
    Kpath_cnstr   = (T2+S-1,) Boolean vector, =True if K_t<=0
    Lpath_new     = (T2+S-1,) vector, new path of the aggregate labor
    rpath_new     = (T2+S-1,) vector, updated time path of interest rate
    wpath_new     = (T2+S-1,) vector, updated time path of the wages
    Ypath         = (T2+S-1,) vector, equilibrium time path of aggregate
                    output (GDP) Y_t
    Cpath         = (T2+S-1,) vector, equilibrium time path of aggregate
                    consumption C_t
    RCerrPath     = (T2+S-2,) vector, equilibrium time path of the
                    resource constraint error:
                    Y_t - C_t - K_{t+1} + (1-delta)*K_t
    KL_path_new   = (2*T2,) vector, appended K_path_new and L_path_new
                    from observation 1 to T2
    KL_path_init  = (2*T2,) vector, appended K_path_init and L_path_init
                    from observation 1 to T2
    Kpath         = (T2+S-1,) vector, equilibrium time path of aggregate
                    capital stock K_t
    Lpath         = (T2+S-1,) vector, equilibrium time path of aggregate
                    labor L_t
    tpi_time      = scalar, time to compute TPI solution (seconds)
    tpi_output    = length 14 dictionary, {cpath, npath, bpath, wpath,
                    rpath, Kpath, Lpath, Ypath, Cpath, bSp1_err_path,
                    n_err_path, b_err_path, RCerrPath, tpi_time}

    FILES CREATED BY THIS FUNCTION:
        Kpath.png
        Lpath.png
        Ypath.png
        C_aggr_path.png
        wpath.png
        rpath.png
        cpath.png
        npath.png
        bpath.png

    RETURNS: tpi_output
    --------------------------------------------------------------------
    '''
    start_time = time.clock()
    (S, T1, T2, beta, sigma, l_tilde, b_ellip, upsilon, chi_n_vec, A,
        alpha, delta, r_star, K_s_ss, L_ss, C_ss, b_splus1_ss, n_ss, maxiter,
        mindist, TPI_tol, xi, diff, hh_fsolve) = params

    # Make arrays of initial guesses for labor supply and savings
    domain = np.linspace(0, T2, T2)
    domain2 = np.tile(domain.reshape(T2, 1), (1, S))
    ending_b = b_splus1_ss
    initial_b = bvec1
    guesses_b = (-1 / (domain2 + 1)) * (ending_b - initial_b) + ending_b
    ending_b_tail = np.tile(ending_b.reshape(1, S), (S, 1))
    guesses_b = np.append(guesses_b, ending_b_tail, axis=0)

    domain3 = np.tile(np.linspace(0, 1, T2).reshape(T2, 1,), (1, S))
    initial_n = n_ss
    guesses_n = domain3 * (n_ss - initial_n) + initial_n
    ending_n_tail = np.tile(n_ss.reshape(1, S), (S, 1))
    guesses_n = np.append(guesses_n, ending_n_tail, axis=0)
    guesses = (guesses_b, guesses_n)

    # With an open economy, the entire path of interest rates is exogenous and
    # this path of interest rates exactly pins down the wage rates - which
    # will therefore also be constant over time.
    rpath = r_star *np.ones(T2 + S - 1)
    w_params = (A, alpha, delta)
    wpath = firms.get_w(rpath, w_params)
    K_params = (A, alpha, delta)
    Y_params = (A, alpha)
    cnb_params = (S, T2, beta, sigma, l_tilde, b_ellip, upsilon,
                  chi_n_vec, bvec1, TPI_tol, diff)

    # if hh_fsolve:
    #     ind = np.arange(S)
    #     bn_args = (rpath, wpath, ind, S, T2, beta, sigma, l_tilde, b_ellip,
    #                upsilon, chi_n_vec, bvec1, TPI_tol, diff)
    #     npath, b_splus1_path, n_err_path, b_err_path = solve_bn_path(guesses, bn_args)
    #     bSp1_err_path = np.zeros((S,T2 + S - 1))
    #     b_s_path = np.zeros((S,T2 + S -1))
    #     b_s_path[:,0] = bvec1
    #     b_s_path[1:,1:] = b_splus1_path[:-1,:-1]
    #     cpath = hh.get_cons(rpath, wpath, b_s_path, b_splus1_path, npath)
    # else:
    #     cpath, npath, b_s_path, n_err_path, b_err_path, bSp1_err_path = \
    #         get_cnbpath(cnb_params, rpath, wpath)
    #     b_splus1_path = np.append(b_s_path[1:,:T2],
    #                               np.reshape(bSp1_err_path[-1,:],(1,T2))
    #                               ,axis=0)

    if hh_fsolve:
        ind = np.arange(S)
        bn_args = (rpath, wpath, ind, S, T2, beta, sigma, l_tilde, b_ellip,
                   upsilon, chi_n_vec, bvec1, TPI_tol, diff)
        npath, b_splus1_path, n_err_path, b_err_path = solve_bn_path(guesses, bn_args)
        bSp1_err_path = np.zeros((S,T2 + S - 1))
        b_s_path = np.zeros((S,T2 + S -1))
        b_s_path[:,0] = bvec1
        b_s_path[1:,1:] = b_splus1_path[:-1,:-1]
        cpath = hh.get_cons(rpath, wpath, b_s_path, b_splus1_path, npath)
    else:
        cpath, npath, b_s_path, n_err_path, b_err_path, bSp1_err_path = \
            get_cnbpath(cnb_params, rpath, wpath)
        b_splus1_path = np.append(b_s_path[1:,:T2],
                                  np.reshape(bSp1_err_path[-1,:],(1,T2))
                                  ,axis=0)

    Lpath = aggr.get_L_s(npath)
    Lpath[T2:] = L_ss
    K_s_path, Kpath_cnstr = aggr.get_K_s(b_s_path)
    K_s_path[T2:] = K_s_ss
    K_d_path = firms.get_K_d(rpath, Lpath, K_params)
    Ypath = aggr.get_Y(Y_params, K_d_path, Lpath)
    Cpath = aggr.get_C(cpath)
    Cpath[T2:] = C_ss
    NXpath = (Ypath[:-1] - Cpath[:-1] - K_s_path[1:] +
                 (1 - delta) * K_s_path[:-1])
    RCerrPath = (Ypath[:-2] - Cpath[:-2] - K_s_path[1:-1] +
                 (1 - delta) * K_s_path[:-2] - NXpath[:-1])

    tpi_time = time.clock() - start_time

    tpi_output = {
        'cpath': cpath, 'npath': npath, 'b_s_path': b_s_path,
        'b_splus1_path': b_splus1_path, 'wpath': wpath, 'rpath': rpath,
        'K_d_path': K_d_path, 'K_s_path': K_s_path,
        'Lpath': Lpath, 'Ypath': Ypath, 'Cpath': Cpath,
        'bSp1_err_path': bSp1_err_path, 'n_err_path': n_err_path,
        'b_err_path': b_err_path, 'RCerrPath': RCerrPath, 'tpi_time': tpi_time}

    # Print maximum resource constraint error. Only look at resource
    # constraint up to period T2 - 1 because period T2 includes K_{t+1},
    # which was forced to be the steady-state
    print('Max abs. RC error: ', "%10.4e" %
          (np.absolute(RCerrPath[:T2 - 1]).max()))

    # Print TPI computation time
    utils.print_time(tpi_time, 'TPI')

    if graphs:
        '''
        ----------------------------------------------------------------
        cur_path    = string, path name of current directory
        output_fldr = string, folder in current path to save files
        output_dir  = string, total path of images folder
        output_path = string, path of file name of figure to be saved
        tvec        = (T2+S-1,) vector, time period vector
        tgridT      = (T2,) vector, time period vector from 1 to T2
        sgrid       = (S,) vector, all ages from 1 to S
        tmat        = (S, T2) matrix, time periods for decisions ages
                      (S) and time periods (T2)
        smat        = (S, T2) matrix, ages for all decisions ages (S)
                      and time periods (T2)
        ----------------------------------------------------------------
        '''
        # Create directory if images directory does not already exist
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        # Plot time path of aggregate capital stock
        tvec = np.linspace(1, T2 + S - 1, T2 + S - 1)
        minorLocator = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(tvec, K_s_path, marker='D')
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Time path for aggregate capital stock K')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Aggregate capital $K_{t}$')
        output_path = os.path.join(output_dir, 'Kpath')
        plt.savefig(output_path)
        # plt.show()
        plt.close()

        # Plot time path of aggregate capital stock
        fig, ax = plt.subplots()
        plt.plot(tvec, Lpath, marker='D')
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Time path for aggregate labor L')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Aggregate labor $L_{t}$')
        output_path = os.path.join(output_dir, 'Lpath')
        plt.savefig(output_path)
        # plt.show()
        plt.close()

        # Plot time path of aggregate output (GDP)
        fig, ax = plt.subplots()
        plt.plot(tvec, Ypath, marker='D')
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Time path for aggregate output (GDP) Y')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Aggregate output $Y_{t}$')
        output_path = os.path.join(output_dir, 'Ypath')
        plt.savefig(output_path)
        # plt.show()
        plt.close()

        # Plot time path of aggregate consumption
        fig, ax = plt.subplots()
        plt.plot(tvec, Cpath, marker='D')
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Time path for aggregate consumption C')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Aggregate consumption $C_{t}$')
        output_path = os.path.join(output_dir, 'C_aggr_path')
        plt.savefig(output_path)
        # plt.show()
        plt.close()

        # Plot time path of real wage
        fig, ax = plt.subplots()
        plt.plot(tvec, wpath, marker='D')
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Time path for real wage w')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Real wage $w_{t}$')
        output_path = os.path.join(output_dir, 'wpath')
        plt.savefig(output_path)
        # plt.show()
        plt.close()

        # Plot time path of real interest rate
        fig, ax = plt.subplots()
        plt.plot(tvec, rpath, marker='D')
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Time path for real interest rate r')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Real interest rate $r_{t}$')
        output_path = os.path.join(output_dir, 'rpath')
        plt.savefig(output_path)
        # plt.show()
        plt.close()

        # Plot time path of individual consumption distribution
        tgridT = np.linspace(1, T2, T2)
        sgrid = np.linspace(1, S, S)
        tmat, smat = np.meshgrid(tgridT, sgrid)
        cmap_c = cm.get_cmap('summer')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel(r'period-$t$')
        ax.set_ylabel(r'age-$s$')
        ax.set_zlabel(r'individual consumption $c_{s,t}$')
        strideval = max(int(1), int(round(S / 10)))
        ax.plot_surface(tmat, smat, cpath[:, :T2], rstride=strideval,
                        cstride=strideval, cmap=cmap_c)
        output_path = os.path.join(output_dir, 'cpath')
        plt.savefig(output_path)
        # plt.show()
        plt.close()

        # Plot time path of individual labor supply distribution
        cmap_n = cm.get_cmap('summer')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel(r'period-$t$')
        ax.set_ylabel(r'age-$s$')
        ax.set_zlabel(r'individual labor supply $n_{s,t}$')
        strideval = max(int(1), int(round(S / 10)))
        ax.plot_surface(tmat, smat, npath[:, :T2], rstride=strideval,
                        cstride=strideval, cmap=cmap_n)
        output_path = os.path.join(output_dir, 'npath')
        plt.savefig(output_path)
        # plt.show()
        plt.close()

        # Plot time path of individual savings distribution
        cmap_b = cm.get_cmap('summer')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel(r'period-$t$')
        ax.set_ylabel(r'age-$s$')
        ax.set_zlabel(r'individual savings $b_{s+1,t+1}$')
        strideval = max(int(1), int(round(S / 10)))
        ax.plot_surface(tmat, smat, b_splus1_path[:, :T2], rstride=strideval,
                        cstride=strideval, cmap=cmap_b)
        output_path = os.path.join(output_dir, 'bpath')
        plt.savefig(output_path)
        # plt.show()
        plt.close()

    return tpi_output
