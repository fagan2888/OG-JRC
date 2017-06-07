'''
------------------------------------------------------------------------
This module contains the functions used to solve the steady state of
the model with S-period lived agents and endogenous labor supply and
an unbalanced government budget constraint from Chapter 15 of the OG
textbook.

This Python module imports the following module(s):
    households.py
    firms.py
    aggregates.py
    utilities.py
    tax.py

This Python module defines the following function(s):
    inner_loop()
    get_SS_bsct()
------------------------------------------------------------------------
'''
# Import packages
import time
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import households as hh
import firms, tax
import aggregates as aggr
import utilities as utils

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''

def inner_loop(r, w, Y, x, params):
    '''
    --------------------------------------------------------------------
    Given values for r and w, solve for equilibrium errors from the two
    first order conditions of the firm
    --------------------------------------------------------------------
    INPUTS:
    r      = scalar > 0, guess at steady-state interest rate
    w      = scalar > 0, guess at steady-state wage
    Y      = scalar > 0, guess steady-state output
    x      = scalar > 0, guess as steady-state transfers per household
    params = length 16 tuple, (c1_init, S, beta, sigma, l_tilde, b_ellip,
                               upsilon, chi_n_vec, A, alpha, delta, tax_params,
                               fiscal_params, diff, hh_fsolve, SS_tol)


    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        hh.bn_solve()
        hh.c1_bSp1err()
        hh.get_cnb_vecs()
        aggr.get_K()
        aggr.get_L()
        firms.get_r()
        firms.get_w()

    OBJECTS CREATED WITHIN FUNCTION:
    c1_init       = scalar > 0, initial guess for c1
    S             = integer >= 3, number of periods in individual lifetime
    beta          = scalar in (0,1), discount factor
    sigma         = scalar >= 1, coefficient of relative risk aversion
    l_tilde       = scalar > 0, per-period time endowment for every agent
    b_ellip       = scalar > 0, fitted value of b for elliptical disutility
                    of labor
    upsilon       = scalar > 1, fitted value of upsilon for elliptical
                    disutility of labor
    chi_n_vec     = (S,) vector, values for chi^n_s
    A             = scalar > 0, total factor productivity parameter in
                    firms' production function
    alpha         = scalar in (0,1), capital share of income
    delta         = scalar in [0,1], model-period depreciation rate of
                    capital
    tax_params    = length 3 tuple, (tau_l, tau_k, tau_c)
    fiscal_params = length 7 tuple, (tG1, tG2, alpha_X, alpha_G, rho_G,
                                     alpha_D, alpha_D0)
    diff          = boolean, =True if simple difference Euler errors,
                    otherwise percent deviation Euler errors
    hh_fsolve     = boolean, =True if solve inner-loop household problem by
                    choosing c_1 to set final period savings b_{S+1}=0.
                    Otherwise, solve the household problem as multivariate
                    root finder with 2S-1 unknowns and equations
    SS_tol        = scalar > 0, tolerance level for steady-state fsolve
    tau_l         = scalar, marginal tax rate on labor income
    tau_k         = scalar, marginal tax rate on capital income
    tau_c         = scalar, marginal tax rate on corporate income
    tG1           = integer, model period when budget closure rule begins
    tG2           = integer, model period when budget is closed
    alpha_X       = scalar, ratio of lump sum transfers to GDP
    alpha_G       = scalar, ratio of government spending to GDP prior to
                            budget closure rule beginning
    rho_G         = scalar in (0,1), rate of convergence to SS budget
    alpha_D       = scalar, steady-state debt to GDP ratio
    alpha_D0      = scalar, debt to GDP ratio in the initial period
    r_params      = length 3 tuple, args to pass into get_r()
    w_params      = length 2 tuple, args to pass into get_w()
    Y_params      = length 2 tuple, args to pass into get_Y()
    b_init        = (S-1,) vector, initial guess at distribution of savings
    n_init        = (S,) vector, initial guess at distribution of labor supply
    guesses       = (2S-1,) vector, initial guesses at b and n
    bn_params     = length 12 tuple, parameters to pass to solve_bn_path()
                    (r, w, x, S, beta, sigma, l_tilde,
                        b_ellip, upsilon, chi_n_vec, tax_params, diff)
    euler_errors  = (2S-1,) vector, Euler errors for FOCs for b and n
    b_splus1_vec  = (S,) vector, optimal savings choice
    nvec          = (S,) vector, optimal labor supply choice
    b_Sp1         = scalar, last period savings
    b_s_vec       = (S,) vector, wealth enter period with
    cvec          = (S,) vector, optimal consumption
    rpath         = (S,) vector, lifetime path of interest rates
    wpath         = (S,) vector, lifetime path of wages
    c1_args       = length 10 tuple, args to pass into c1_bSp1err()
    c1_options    = length 1 dict, options for c1_bSp1err()
    results_c1    = results object, results from c1_bSp1err()
    c1            = scalar > 0, optimal initial period consumption given r
                    and w
    cnb_args      = length 8 tuple, args to pass into get_cnb_vecs()
    cvec          = (S,) vector, lifetime consumption (c1, c2, ...cS)
    nvec          = (S,) vector, lifetime labor supply (n1, n2, ...nS)
    bvec          = (S,) vector, lifetime savings (b1, b2, ...bS) with b1=0
    b_Sp1         = scalar, final period savings, should be close to zero
    B             = scalar > 0, aggregate savings
    B_cnstr       = boolean, =True if B < 0
    L             = scalar > 0, aggregate labor
    debt          = scalar, total government debt
    K             = scalar > 0, aggregate capital stock
    K_cnstr       = boolean, =True if K < 0
    r_new         = scalar > 0, implied steady-state interest rate
    w_new         = scalar > 0, implied steady-state wage
    Y_new         = scalar >0, implied steady-state output
    x_new         = scalar >=0, implied transfers per household


    FILES CREATED BY THIS FUNCTION: None

    RETURNS: B, K, L, Y_new, debt, cvec, nvec, b_s_vec, b_splus1_vec,
                b_Sp1, x_new, r_new, w_new
    --------------------------------------------------------------------
    '''
    c1_init, S, beta, sigma, l_tilde, b_ellip, upsilon,\
        chi_n_vec, A, alpha, delta, tax_params, fiscal_params,\
        diff, hh_fsolve, SS_tol = params
    tau_l, tau_k, tau_c = tax_params
    tG1, tG2, alpha_X, alpha_G, rho_G, alpha_D, alpha_D0 = fiscal_params

    r_params = (A, alpha, delta, tau_c)
    w_params = (A, alpha)
    Y_params = (A, alpha)

    if hh_fsolve:
        b_init = np.ones((S - 1, 1)) * 0.05
        n_init = np.ones((S, 1)) * 0.4
        guesses = np.append(b_init, n_init)
        bn_params = (r, w, x, S, beta, sigma, l_tilde,
                     b_ellip, upsilon, chi_n_vec, tax_params, diff)
        [solutions, infodict, ier, message] = \
            opt.fsolve(hh.bn_solve, guesses, args=bn_params,
                       xtol=SS_tol, full_output=True)
        euler_errors = infodict['fvec']
        print('Max Euler errors: ',
              np.absolute(euler_errors).max())
        b_splus1_vec = np.append(solutions[:S - 1], 0.0)
        nvec = solutions[S - 1:]
        b_Sp1 = 0.0
        b_s_vec = np.append(0.0, b_splus1_vec[:-1])
        cvec = hh.get_cons(r, w, b_s_vec, b_splus1_vec,
                               nvec, x, tax_params)
    else:
        rpath = r * np.ones(S)
        wpath = w * np.ones(S)
        xpath = x * np.ones(S)
        c1_options = {'maxiter': 500}
        c1_args = (0.0, beta, sigma, l_tilde, b_ellip, upsilon,
                   chi_n_vec, tax_params, xpath, rpath, wpath, diff)
        results_c1 = \
            opt.root(hh.c1_bSp1err, c1_init, args=(c1_args),
                     method='lm', tol=SS_tol,
                     options=(c1_options))
        c1_new = results_c1.x
        cnb_args = (0.0, beta, sigma, l_tilde, b_ellip, upsilon, chi_n_vec,
                    tax_params, diff)
        cvec, nvec, b_s_vec, b_Sp1 = \
            hh.get_cnb_vecs(c1_new, rpath, wpath, xpath, cnb_args)
        b_splus1_vec = np.append(b_s_vec[1:], b_Sp1)

    B, B_cnstr = aggr.get_K(b_s_vec)
    L = aggr.get_L(nvec)
    L = np.maximum(0.0001, L)
    debt = alpha_D*Y
    K = B - debt
    K_cnstr = K < 0
    if K_cnstr:
        print('Aggregate capital constraint is violated K<=0 for ' +
              'in the steady state.')
    r_new = firms.get_r(r_params, K, L)
    w_new = firms.get_w(w_params, K, L)
    Y_new = aggr.get_Y(Y_params, K, L)
    x_new = (alpha_X*Y_new)/S


    return B, K, L, Y_new, debt, cvec, nvec, b_s_vec, b_splus1_vec, \
                b_Sp1, x_new, r_new, w_new

def get_SS_bsct(init_vals, args, graphs=False):
    '''
    --------------------------------------------------------------------
    Solve for the steady-state solution of the S-period-lived agent OG
    model with endogenous labor supply using the bisection method for
    the outer loop
    --------------------------------------------------------------------
    INPUTS:
    init_vals = length 5 tuple,
                (Kss_init, Lss_init, rss_init, wss_init,c1_init)
    args      = length 16 tuple, (S, beta, sigma, l_tilde, b_ellip,
                upsilon, chi_n_vec, A, alpha, delta, tax_params,
                fiscal_params, SS_tol, EulDiff, hh_fsolve, KL_outer)
    graphs    = boolean, =True if output steady-state graphs

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        firms.get_r()
        firms.get_w()
        hh.bn_solve()
        hh.c1_bSp1err()
        hh.get_cnb_vecs()
        aggr.get_K()
        aggr.get_L()
        aggr.get_Y()
        aggr.get_C()
        hh.get_cons()
        hh.get_n_errors()
        hh.get_b_errors()
        utils.print_time()

    OBJECTS CREATED WITHIN FUNCTION:
    start_time       = scalar > 0, clock time at beginning of program
    Kss_init         = scalar > 0, initial guess for steady-state aggregate
                       capital stock
    Lss_init         = scalar > 0, initial guess for steady-state aggregate
                       labor
    rss_init         = scalar > 0, initial guess for steady-state interest
                       rate
    wss_init         = scalar > 0, initial guess for steady-state wage
    c1_init          = scalar > 0, initial guess for first period consumpt'n
    S                = integer in [3, 80], number of periods an individual
                       lives
    beta             = scalar in (0,1), discount factor for each model per
    sigma            = scalar > 0, coefficient of relative risk aversion
    l_tilde          = scalar > 0, time endowment for each agent each period
    b_ellip          = scalar > 0, fitted value of b for elliptical
                       disutility of labor
    upsilon          = scalar > 1, fitted value of upsilon for elliptical
                       disutility of labor
    chi_n_vec        = (S,) vector, values for chi^n_s
    A                = scalar > 0, total factor productivity parameter in
                       firms' production function
    alpha            = scalar in (0,1), capital share of income
    delta            = scalar in [0,1], model-period depreciation rate of
                       capital
    tax_params       = length 3 tuple, (tau_l, tau_k, tau_c)
    fiscal_params    = length 7 tuple, (tG1, tG2, alpha_X, alpha_G, rho_G,
                                     alpha_D, alpha_D0)
    SS_tol           = scalar > 0, tolerance level for steady-state fsolve
    EulDiff          = Boolean, =True if want difference version of Euler
                       errors beta*(1+r)*u'(c2) - u'(c1), =False if want
                       ratio version [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
    hh_fsolve        = Boolean, =True if want to solve HH problem with one
                       large root finder call
    tau_l            = scalar, marginal tax rate on labor income
    tau_k            = scalar, marginal tax rate on capital income
    tau_c            = scalar, marginal tax rate on corporate income
    tG1              = integer, model period when budget closure rule begins
    tG2              = integer, model period when budget is closed
    alpha_X          = scalar, ratio of lump sum transfers to GDP
    alpha_G          = scalar, ratio of government spending to GDP prior to
                            budget closure rule beginning
    rho_G            = scalar in (0,1), rate of convergence to SS budget
    alpha_D          = scalar, steady-state debt to GDP ratio
    alpha_D0         = scalar, debt to GDP ratio in the initial period
    maxiter_SS       = integer >= 1, maximum number of iterations in outer
                       loop bisection method
    iter_SS          = integer >= 0, index of iteration number
    mindist_SS       = scalar > 0, minimum distance tolerance for
                       convergence
    dist_SS          = scalar > 0, distance metric for current iteration
    xi_SS            = scalar in (0,1], updating parameter
    KL_init          = (2,) vector, (K_init, L_init)
    c1_options       = length 1 dict, options to pass into
                       opt.root(c1_bSp1err,...)
    cnb_args         = length 9 tuple, args to pass into get_cnb_vecs()
    r_params         = length 3 tuple, args to pass into get_r()
    w_params         = length 2 tuple, args to pass into get_w()
    Y_params         = length 2 tuple, args to pass into get_Y()
    K_init           = scalar, initial value of aggregate capital stock
    L_init           = scalar, initial value of aggregate labor
    r_init           = scalar, initial value for interest rate
    w_init           = scalar, initial value for wage
    Y_init           = scalar, initial value for output
    x_init           = scalar, initial value for per household lump sum transfers
    rpath            = (S,) vector, lifetime path of interest rates
    wpath            = (S,) vector, lifetime path of wages
    xpath            = (S,) vector, lifetime path of lump sum transfers
    c1_args          = length 10 tuple, args to pass into c1_bSp1err()
    results_c1       = results object, root finder results from
                       opt.root(c1_bSp1err,...)
    c1_new           = scalar, updated value of optimal c1 given r_init and
                       w_init
    cvec_new         = (S,) vector, updated values for lifetime consumption
    nvec_new         = (S,) vector, updated values for lifetime labor supply
    b_splus1_vec_new = (S,) vector, updated values for lifetime savings
                       (b1, b2,...bS)
    b_s_vec_new      = (S,) vector, updated values for lifetime savings enter
                       period with (b0, b1,...bS)
    b_Sp1_new        = scalar, updated value for savings in last period,
                       should be arbitrarily close to zero
    B_new            = scalar, aggregate household savings given bvec_new
    B_cnstr          = boolean, =True if K_new <= 0
    L_new            = scalar, updated L given nvec_new
    debt_ss          = scalar, government debt in the SS
    K_new            = scalar, updated K given bvec_new and SS debt
    K_cnstr          = boolean, =True if K_new <= 0
    KL_new           = (2,) vector, updated K and L given bvec_new, nvec_new
    K_ss             = scalar > 0, steady-state aggregate capital stock
    L_ss             = scalar > 0, steady-state aggregate labor
    B_ss             = scalar > 0, steady-state aggregate savings
    r_ss             = scalar > 0, steady-state interest rate
    w_ss             = scalar > 0, steady-state wage
    x_ss             = scalar > 0, steady-state per household lump sum transfers
    c1_ss            = scalar > 0, steady-state consumption in first period
    c_ss             = (S,) vector, steady-state lifetime consumption
    n_ss             = (S,) vector, steady-state lifetime labor supply
    b_splus1_ss      = (S,) vector, steady-state lifetime savings
                       (b1_ss, b2_ss, ...bS+1_ss)
    b_s_ss           = (S,) vector, steady-state lifetime savings enter period with
                       (b0_ss, b2_ss, ...bS_ss) where b0_ss=0
    b_Sp1_ss         = scalar, steady-state savings for period after last
                       period of life. b_Sp1_ss approx. 0 in equilibrium
    Y_ss             = scalar > 0, steady-state aggregate output (GDP)
    C_ss             = scalar > 0, steady-state aggregate consumption
    n_err_params     = length 5 tuple, args to pass into get_n_errors()
    n_err_ss         = (S,) vector, lifetime labor supply Euler errors
    b_err_params     = length 2 tuple, args to pass into get_b_errors()
    b_err_ss         = (S-1) vector, lifetime savings Euler errors
    rev_params       = length 4 tuple, (A, alpha, delta, tax_params)
    R_ss             = scalar, steady-state tax revenue
    X_ss             = scalar, total steady-state government transfers
    debt_service_ss  = scalar, steady-state debt service cost
    G_ss             = steady-state government spending
    RCerr_ss         = scalar, resource constraint error
    ss_time          = scalar, seconds elapsed to run steady-state comput'n
    ss_output        = length 17 dict, steady-state objects {n_ss, b_s_ss,
                       b_splus1_ss, c_ss, b_Sp1_ss, w_ss, r_ss, B_ss, K_ss,
                       L_ss, Y_ss, C_ss, G_ss, n_err_ss, b_err_ss, RCerr_ss,
                       ss_time}


    FILES CREATED BY THIS FUNCTION:
        SS_bc.png
        SS_n.png

    RETURNS: ss_output
    --------------------------------------------------------------------
    '''
    start_time = time.clock()
    Kss_init, Lss_init, rss_init, wss_init, c1_init = init_vals
    (S, beta, sigma, l_tilde, b_ellip, upsilon, chi_n_vec, A, alpha,
        delta, tax_params, fiscal_params, SS_tol, EulDiff, hh_fsolve) = args
    tau_l, tau_k, tau_c = tax_params
    tG1, tG2, alpha_X, alpha_G, rho_G, alpha_D, alpha_D0 = fiscal_params
    maxiter_SS = 200
    iter_SS = 0
    mindist_SS = 1e-10
    dist_SS = 10
    xi_SS = 0.15
    KL_init = np.array([Kss_init, Lss_init])
    r_params = (A, alpha, delta, tau_c)
    w_params = (A, alpha)
    Y_params = (A, alpha)
    inner_loop_params = (c1_init, S, beta, sigma, l_tilde, \
                         b_ellip, upsilon, chi_n_vec, A, alpha, delta,\
                         tax_params, fiscal_params,\
                         EulDiff, hh_fsolve, SS_tol)
    while (iter_SS < maxiter_SS) and (dist_SS >= mindist_SS):
        iter_SS += 1
        K_init, L_init = KL_init
        r_init = firms.get_r(r_params, K_init, L_init)
        w_init = firms.get_w(w_params, K_init, L_init)
        Y_init = aggr.get_Y(Y_params, K_init, L_init)
        x_init = (alpha_X*Y_init)/S

        B_new, K_new, L_new, Y_new, debt, cvec, nvec, b_s_vec, b_splus1_vec,\
            b_Sp1, x_new, r_new, w_new\
             =  inner_loop(r_init, w_init, Y_init, x_init, inner_loop_params)

        KL_new = np.array([K_new, L_new])
        dist_SS = ((KL_new - KL_init) ** 2).sum()
        KL_init = xi_SS * KL_new + (1 - xi_SS) * KL_init

        rev_params = (A, alpha, delta, tax_params)
        R_new = tax.revenue(r_new, w_new, b_s_vec, nvec, K_new, L_new, rev_params)
        X_new = x_new*S
        G_new = R_new - (X_new + debt*r_new)
        print('tax rev to GDP: ', R_new/Y_new)
        print('SS outlays to GDP: ', ((debt*r_new)+X_new+G_new)/Y_new)
        print('SS G spending to GDP: ', G_new/Y_new)
        print('factor prices: ', r_new, w_new)
        print('SS Iteration=', iter_SS, ', SS Distance=', dist_SS)

    B_ss, K_ss, L_ss, Y_ss, debt_ss, c_ss, n_ss, b_s_ss, b_splus1_ss,\
        b_Sp1_ss, x_ss, r_ss, w_ss\
         =  inner_loop(r_new, w_new, Y_new, x_new, inner_loop_params)

    C_ss = aggr.get_C(c_ss)
    n_err_args = (w_ss, c_ss, sigma, l_tilde, chi_n_vec, b_ellip, upsilon,
                  tau_l, EulDiff)
    n_err_ss = hh.get_n_errors(n_ss, n_err_args)
    b_err_params = (beta, sigma, tau_k)
    b_err_ss = hh.get_b_errors(b_err_params, r_ss, c_ss, EulDiff)
    rev_params = (A, alpha, delta, tax_params)
    R_ss = tax.revenue(r_ss, w_ss, b_s_ss, n_ss, K_ss, L_ss, rev_params)
    X_ss = x_ss*S
    debt_service_ss = r_ss*alpha_D*Y_ss
    G_ss = R_ss - (X_ss + debt_service_ss)
    RCerr_ss = Y_ss - C_ss - delta * K_ss - G_ss

    print('SS tax rev to GDP: ', R_ss/Y_ss)
    print('SS outlays to GDP: ', (debt_service_ss+X_ss+G_ss)/Y_ss)
    print('SS G spending to GDP: ', G_ss/Y_ss)

    ss_time = time.clock() - start_time

    ss_output = {
        'n_ss': n_ss, 'b_s_ss': b_s_ss, 'b_splus1_ss': b_splus1_ss,
        'c_ss': c_ss, 'b_Sp1_ss': b_Sp1_ss, 'w_ss': w_ss, 'r_ss': r_ss,
        'B_ss': B_ss, 'K_ss': K_ss, 'L_ss': L_ss,
        'Y_ss': Y_ss, 'C_ss': C_ss, 'G_ss': G_ss, 'X_ss': X_ss,
        'n_err_ss': n_err_ss,
        'b_err_ss': b_err_ss, 'RCerr_ss': RCerr_ss, 'ss_time': ss_time}
    print('n_ss is: ', n_ss)
    print('b_s_ss is: ', b_s_ss)
    print('K_ss=', K_ss, ', L_ss=', L_ss)
    print('r_ss=', r_ss, ', w_ss=', w_ss)
    print('Maximum abs. labor supply Euler error is: ',
          np.absolute(n_err_ss).max())
    print('Maximum abs. savings Euler error is: ',
          np.absolute(b_err_ss).max())
    print('Resource constraint error is: ', RCerr_ss)
    print('Steay-state government spending is: ', G_ss)
    if G_ss < 0:
        print('WARNING: SS debt to GDP ratio and tax policy are generating ' +
              'negative government spending.')
    print('Steay-state tax revenue, transfers, and debt service are: ',
          R_ss, X_ss, debt_service_ss)
    print('Steady-state residual savings b_Sp1 is: ', b_Sp1_ss)

    # Print SS computation time
    utils.print_time(ss_time, 'SS')

    if graphs:
        '''
        ----------------------------------------------------------------
        cur_path    = string, path name of current directory
        output_fldr = string, folder in current path to save files
        output_dir  = string, total path of images folder
        output_path = string, path of file name of figure to be saved
        age_pers    = (S,) vector, ages from 1 to S
        ----------------------------------------------------------------
        '''
        # Create directory if images directory does not already exist
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = 'images'
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        # Plot steady-state consumption and savings distributions
        age_pers = np.arange(1, S + 1)
        fig, ax = plt.subplots()
        plt.plot(age_pers, c_ss, marker='D', label='Consumption')
        plt.plot(age_pers, b_splus1_ss, marker='D', label='Savings')
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        # plt.title('Steady-state consumption and savings', fontsize=20)
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'Units of consumption')
        plt.xlim((0, S + 1))
        # plt.ylim((-1.0, 1.15 * (b_ss.max())))
        plt.legend(loc='upper left')
        output_path = os.path.join(output_dir, 'SS_bc')
        plt.savefig(output_path)
        # plt.show()
        plt.close()

        # Plot steady-state labor supply distributions
        fig, ax = plt.subplots()
        plt.plot(age_pers, n_ss, marker='D', label='Labor supply')
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        # plt.title('Steady-state labor supply', fontsize=20)
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'Labor supply')
        plt.xlim((0, S + 1))
        # plt.ylim((-0.1, 1.15 * (n_ss.max())))
        plt.legend(loc='upper right')
        output_path = os.path.join(output_dir, 'SS_n')
        plt.savefig(output_path)
        # plt.show()
        plt.close()

    return ss_output
