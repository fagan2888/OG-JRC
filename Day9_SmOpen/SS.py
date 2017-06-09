'''
------------------------------------------------------------------------
This module contains the functions used to solve the steady state of
the model with S-period lived agents and endogenous labor supply from
Chapter 7 of the OG textbook.

This Python module imports the following module(s):
    households.py
    firms.py
    aggregates.py
    utilities.py

This Python module defines the following function(s):
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
import firms
import aggregates as aggr
import utilities as utils

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''




def get_SS(init_vals, args, graphs=False):
    '''
    --------------------------------------------------------------------
    Solve for the steady-state solution of the S-period-lived agent OG
    model with endogenous labor supply and a small open economy.
    --------------------------------------------------------------------
    INPUTS:
    init_vals = length 5 tuple,
                (Kss_init, Lss_init, rss_init, wss_init,c1_init)
    args      = length 14 tuple, (S, beta, sigma, l_tilde, b_ellip,
                upsilon, chi_n_vec, A, alpha, delta, SS_tol, EulDiff,
                hh_fsolve, KL_outer)
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
    start_time   = scalar > 0, clock time at beginning of program
    Kss_init     = scalar > 0, initial guess for steady-state aggregate
                   capital stock supplied
    Lss_init     = scalar > 0, initial guess for steady-state aggregate
                   labor
    rss_init     = scalar > 0, initial guess for steady-state interest
                   rate
    wss_init     = scalar > 0, initial guess for steady-state wage
    c1_init      = scalar > 0, initial guess for first period consumpt'n
    S            = integer in [3, 80], number of periods an individual
                   lives
    beta         = scalar in (0,1), discount factor for each model per
    sigma        = scalar > 0, coefficient of relative risk aversion
    l_tilde      = scalar > 0, time endowment for each agent each period
    b_ellip      = scalar > 0, fitted value of b for elliptical
                   disutility of labor
    upsilon      = scalar > 1, fitted value of upsilon for elliptical
                   disutility of labor
    chi_n_vec    = (S,) vector, values for chi^n_s
    A            = scalar > 0, total factor productivity parameter in
                   firms' production function
    alpha        = scalar in (0,1), capital share of income
    delta        = scalar in [0,1], model-period depreciation rate of
                   capital
    SS_tol       = scalar > 0, tolerance level for steady-state fsolve
    EulDiff      = Boolean, =True if want difference version of Euler
                   errors beta*(1+r)*u'(c2) - u'(c1), =False if want
                   ratio version [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
    maxiter_SS   = integer >= 1, maximum number of iterations in outer
                   loop bisection method
    iter_SS      = integer >= 0, index of iteration number
    mindist_SS   = scalar > 0, minimum distance tolerance for
                   convergence
    dist_SS      = scalar > 0, distance metric for current iteration
    xi_SS        = scalar in (0,1], updating parameter
    KL_init      = (2,) vector, (K_init, L_init)
    c1_options   = length 1 dict, options to pass into
                   opt.root(c1_bSp1err,...)
    cnb_args     = length 8 tuple, args to pass into get_cnb_vecs()
    #r_params     = length 3 tuple, args to pass into get_r()
    w_params     = length 2 tuple, args to pass into get_w()
    K_init       = scalar, initial value of aggregate capital stock supplied
    L_init       = scalar, initial value of aggregate labor
    r_init       = scalar, initial value for interest rate
    w_init       = scalar, initial value for wage
    K_d          = scalar, capital demand
    rpath        = (S,) vector, lifetime path of interest rates
    wpath        = (S,) vector, lifetime path of wages
    c1_args      = length 10 tuple, args to pass into c1_bSp1err()
    results_c1   = results object, root finder results from
                   opt.root(c1_bSp1err,...)
    c1_new       = scalar, updated value of optimal c1 given r_init and
                   w_init
    cvec_new     = (S,) vector, updated values for lifetime consumption
    nvec_new     = (S,) vector, updated values for lifetime labor supply
    b_s_new      = (S,) vector, updated values for lifetime wealth
    b_splus1_new = (S,) vector, updated values for lifetime savings
                   (b1, b2,...bS)
    b_Sp1_new    = scalar, updated value for savings in last period,
                   should be arbitrarily close to zero
    K_new        = scalar, updated K given bvec_new
    K_cnstr      = boolean, =True if K_new <= 0
    L_new        = scalar, updated L given nvec_new
    KL_new       = (2,) vector, updated K and L given bvec_new, nvec_new
    K_ss         = scalar > 0, steady-state aggregate capital stock supplied
    K_d_ss       = scalar > 0, steady-state aggregate capital stock demanded
    L_ss         = scalar > 0, steady-state aggregate labor
    r_ss         = scalar > 0, steady-state interest rate
    w_ss         = scalar > 0, steady-state wage
    c1_ss        = scalar > 0, steady-state consumption in first period
    c_ss         = (S,) vector, steady-state lifetime consumption
    n_ss         = (S,) vector, steady-state lifetime labor supply
    b_s_ss       = (S,) vector, steady-state wealth enter period with
    b_splus1_ss  = (S,) vector, steady-state lifetime savings
                   (b1_ss, b2_ss, ...bS_ss) where b1_ss=0
    b_Sp1_ss     = scalar, steady-state savings for period after last
                   period of life. b_Sp1_ss approx. 0 in equilibrium
    Y_params     = length 2 tuple, (A, alpha)
    Y_ss         = scalar > 0, steady-state aggregate output (GDP)
    C_ss         = scalar > 0, steady-state aggregate consumption
    n_err_params = length 5 tuple, args to pass into get_n_errors()
    n_err_ss     = (S,) vector, lifetime labor supply Euler errors
    b_err_params = length 2 tuple, args to pass into get_b_errors()
    b_err_ss     = (S-1) vector, lifetime savings Euler errors
    RCerr_ss     = scalar, resource constraint error
    ss_time      = scalar, seconds elapsed to run steady-state comput'n
    ss_output    = length 14 dict, steady-state objects {n_ss, b_ss,
                   c_ss, b_Sp1_ss, w_ss, r_ss, K_ss, L_ss, Y_ss, C_ss,
                   n_err_ss, b_err_ss, RCerr_ss, ss_time}

    FILES CREATED BY THIS FUNCTION:
        SS_bc.png
        SS_n.png

    RETURNS: ss_output
    --------------------------------------------------------------------
    '''
    start_time = time.clock()
    c1_init = init_vals
    (S, beta, sigma, l_tilde, b_ellip, upsilon, chi_n_vec, A, alpha,
        delta, r_star, SS_tol, EulDiff, hh_fsolve) = args
    c1_options = {'maxiter': 500}
    cnb_args = (0.0, beta, sigma, l_tilde, b_ellip, upsilon, chi_n_vec,
                EulDiff)
    w_params = (A, alpha, delta)
    K_params = (A, alpha, delta)
    r_ss = r_star
    w_ss = firms.get_w(r_ss, w_params)

    if hh_fsolve:
        b_init = np.ones((S - 1, 1)) * 0.05
        n_init = np.ones((S, 1)) * 0.4
        guesses = np.append(b_init, n_init)
        bn_params = (r_ss, w_ss, S, beta, sigma, l_tilde, b_ellip,
                     upsilon, chi_n_vec, EulDiff)
        [solutions, infodict, ier, message] = \
            opt.fsolve(hh.bn_solve, guesses, args=bn_params,
                       xtol=SS_tol, full_output=True)
        euler_errors = infodict['fvec']
        print('Max Euler errors: ', np.absolute(euler_errors).max())
        b_splus1_ss = np.append(solutions[:S - 1], 0.0)
        n_ss = solutions[S - 1:]
        b_Sp1_ss = 0.0
        b_s_ss = np.append(0.0, b_splus1_ss[:-1])
        c_ss = hh.get_cons(r_ss, w_ss, b_s_ss, b_splus1_ss, n_ss)
    else:
        rpath = r_ss * np.ones(S)
        wpath = w_ss * np.ones(S)
        c1_args = (0.0, beta, sigma, l_tilde, b_ellip, upsilon,
                   chi_n_vec, rpath, wpath, EulDiff)
        c1_options = {'maxiter': 500}
        results_c1 = \
            opt.root(hh.c1_bSp1err, c1_new, args=(c1_args),
                     method='lm', tol=SS_tol, options=(c1_options))
        c1_ss = results_c1.x
        cnb_args = (0.0, beta, sigma, l_tilde, b_ellip, upsilon,
                    chi_n_vec, EulDiff)
        c_ss, n_ss, b_s_ss, b_Sp1_ss = hh.get_cnb_vecs(c1_ss, rpath,
                                                     wpath, cnb_args)
        b_splus1_ss = np.append(b_s_ss[1:],b_Sp1_ss)

    L_ss = aggr.get_L_s(n_ss)
    K_s_ss, K_cnstr = aggr.get_K_s(b_s_ss)
    K_d_ss = firms.get_K_d(r_ss, L_ss, K_params)
    Y_params = (A, alpha)
    Y_ss = aggr.get_Y(Y_params, K_d_ss, L_ss)
    C_ss = aggr.get_C(c_ss)
    n_err_args = (w_ss, c_ss, sigma, l_tilde, chi_n_vec, b_ellip, upsilon, EulDiff)
    n_err_ss = hh.get_n_errors(n_ss, n_err_args)
    b_err_params = (beta, sigma)
    b_err_ss = hh.get_b_errors(b_err_params, r_ss, c_ss, EulDiff)
    NX_ss = Y_ss - C_ss - delta*K_s_ss
    RCerr_ss = Y_ss - C_ss - delta*K_s_ss - NX_ss

    ss_time = time.clock() - start_time

    ss_output = {
        'n_ss': n_ss, 'b_s_ss': b_s_ss, 'b_splus1_ss': b_splus1_ss,
        'c_ss': c_ss, 'b_Sp1_ss': b_Sp1_ss, 'w_ss': w_ss, 'r_ss': r_ss,
        'K_s_ss': K_s_ss, 'K_d_ss': K_d_ss, 'L_ss': L_ss,
        'Y_ss': Y_ss, 'C_ss': C_ss, 'n_err_ss': n_err_ss,
        'b_err_ss': b_err_ss, 'RCerr_ss': RCerr_ss, 'ss_time': ss_time}
    print('n_ss is: ', n_ss)
    print('b_splus1_ss is: ', b_splus1_ss)
    print('K_s_ss=', K_s_ss, 'K_d_ss=', K_d_ss, ', L_ss=', L_ss)
    print('r_ss=', r_ss, ', w_ss=', w_ss)
    print('Maximum abs. labor supply Euler error is: ',
          np.absolute(n_err_ss).max())
    print('Maximum abs. savings Euler error is: ',
          np.absolute(b_err_ss).max())
    print('Resource constraint error is: ', RCerr_ss)
    print('Net Exports = ', NX_ss)
    print('Output and consumption: ', Y_ss, C_ss)
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
