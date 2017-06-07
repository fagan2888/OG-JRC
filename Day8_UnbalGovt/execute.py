'''
------------------------------------------------------------------------
This program runs the steady state solver as well as the time path
iteration solution for the model with S-period lived agents and
endogenous labor and an unbalanced government budget constraint from
Chapter 15 of the OG textbook.

This Python script imports the following module(s):
    SS.py
    TPI.py
    aggregates.py
    elliputil.py
    utilities.py

This Python script calls the following function(s):
    elp.fit_ellip_CFE()
    ss.get_SS_root()
    ss.get_SS_bsct()
    utils.compare_args()
    aggr.get_K()
    tpi.get_TPI()

Files created by this script:
    OUTPUT/SS/ss_vars.pkl
    OUTPUT/SS/ss_args.pkl
    OUTPUT/TPI/tpi_vars.pkl
    OUTPUT/TPI/tpi_args.pkl
------------------------------------------------------------------------
'''
# Import packages
import numpy as np
import pickle
import os
import SS as ss
import TPI as tpi
import aggregates as aggr
import elliputil as elp
import utilities as utils

'''
------------------------------------------------------------------------
Declare parameters
------------------------------------------------------------------------
S             = integer in [3,80], number of periods an individual lives
beta_annual   = scalar in (0,1), discount factor for one year
beta          = scalar in (0,1), discount factor for each model period
sigma         = scalar > 0, coefficient of relative risk aversion
l_tilde       = scalar > 0, per-period time endowment for every agent
chi_n_vec     = (S,) vector, values for chi^n_s
A             = scalar > 0, total factor productivity parameter in
                firms' production function
alpha         = scalar in (0,1), capital share of income
delta_annual  = scalar in [0,1], one-year depreciation rate of capital
delta         = scalar in [0,1], model-period depreciation rate of
                capital
tau_l         = scalar, marginal tax rate on labor income
tau_k         = scalar, marginal tax rate on capital income
tau_c         = scalar, marginal tax rate on corporate income
tax_params    = length 3 tuple, (tau_l, tau_k, tau_c)
tG1           = integer, Model period where begin fiscal closure rule
tG2           = integer, Model period where final discrete jump in gov't
                spening rule to achieve SS debt ratio
alpha_T       = scalar in [0,1), Ratio of lump sum government transfer to
                GDP in all model periods.
alpha_G       = scalar in [0,1), Ratio of government spending to GDP for
                model periods t<tG1.
rho_G         = scalar in (0,1), Transition speed change in government spending
                to close budget in model periods [tG1, tG2-1]. A lower rho_G
                => slower convergence.
alpha_D       = scalar, Target steady-state government debt to GDP ratio.
                A government surplus would be a negative number.
alpha_D0      = scalar, First-period government debt to GDP ratio. Savings
                would be a negative number.
fiscal_params = length 7 tuple, (tG1, tG2, alpha_T, alpha_G, rho_G,
                                 alpha_D, alpha_D0)
SS_solve      = boolean, =True if want to solve for steady-state
                solution, otherwise retrieve solutions from pickle
SS_tol        = scalar > 0, tolerance level for steady-state fsolve
SS_graphs     = boolean, =True if want graphs of steady-state objects
SS_EulDiff    = boolean, =True if use simple differences in Euler
                errors. Otherwise, use percent deviation form.
SS_outer_root = boolean, =True if use root finder to solve outer loop
                zeros for rss and wss
KL_outer      = boolean, =True if guess K and L in outer loop.
                Otherwise, guess r and w in outer loop
hh_fsolve_SS  = boolean, =False if solve inner-loop household problem by
                choosing c_1 to set final period savings b_{S+1}=0.
                Otherwise, solve the household problem as multivariate
                root finder with 2S-1 unknowns and equations. Applies solution
                method chosen only to the steady-state.
T1            = integer > S, number of time periods until steady state
                is assumed to be reached
T2            = integer > T1, number of time periods after which steady-
                state is forced in TPI
TPI_solve     = boolean, =True if want to solve TPI after solving SS
hh_fsolve_TPI = boolean, =False if solve inner-loop household problem by
                choosing c_1 to set final period savings b_{S+1}=0.
                Otherwise, solve the household problem as multivariate
                root finder with 2S-1 unknowns and equations. Applies solution
                method chosen only to time path iteration.
TPI_tol       = scalar > 0, tolerance level for fsolve's in TPI
maxiter_TPI   = integer >= 1, Maximum number of iterations for TPI
mindist_TPI   = scalar > 0, Convergence criterion for TPI
xi            = scalar in (0,1], TPI path updating parameter
TPI_graphs    = Boolean, =True if want graphs of TPI objects
TPI_EulDiff   = Boolean, =True if want difference version of Euler
                errors beta*(1+r)*u'(c2) - u'(c1), =False if want ratio
                version [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
------------------------------------------------------------------------
'''
# Household parameters
S = int(80)
beta_annual = 0.96
beta = beta_annual ** (80 / S)
sigma = 2.5
l_tilde = 1.0
chi_n_vec = 1.0 * np.ones(S)
# Firm parameters
A = 1.0
alpha = 0.35
delta_annual = 0.05
delta = 1 - ((1 - delta_annual) ** (80 / S))
# Tax parameters
tau_l = 0.25 # linear rate on labor income
tau_k = 0.3 #linear rate on capital income
tau_c = 0.15 #corporate income tax rate
tax_params = (tau_l, tau_k, tau_c)
# Fiscal imbalance parameters.
tG1      = 20
tG2      = int(2*S)
alpha_X  = 0.15#0.10 #0.09
alpha_G  = 0.15#0.12 #0.05
rho_G = 0.05
alpha_D = 0.4
alpha_D0 = 0.59
fiscal_params = (tG1, tG2, alpha_X, alpha_G, rho_G, alpha_D, alpha_D0)

# SS parameters
SS_solve = True
SS_tol = 1e-13
SS_graphs = True
SS_EulDiff = True
hh_fsolve_SS = False
# TPI parameters
T1 = int(round(3.0 * S))
T2 = int(round(3.5 * S))
TPI_solve = True
hh_fsolve_TPI = True
TPI_tol = 1e-13
maxiter_TPI = 200
mindist_TPI = 1e-13
xi = 0.20
TPI_graphs = True
TPI_EulDiff = True

'''
------------------------------------------------------------------------
Fit elliptical utility function to constant Frisch elasticity (CFE)
disutility of labor function by matching marginal utilities along the
support of leisure
------------------------------------------------------------------------
ellip_graph  = Boolean, =True if want to save plot of fit
b_ellip_init = scalar > 0, initial guess for b
upsilon_init = scalar > 1, initial guess for upsilon
ellip_init   = (2,) vector, initial guesses for b and upsilon
Frisch_elast = scalar > 0, Frisch elasticity of labor supply for CFE
               disutility of labor
CFE_scale    = scalar > 0, scale parameter for CFE disutility of labor
cfe_params   = (2,) vector, values for (Frisch, CFE_scale)
b_ellip      = scalar > 0, fitted value of b for elliptical disutility
               of labor
upsilon      = scalar > 1, fitted value of upsilon for elliptical
               disutility of labor
------------------------------------------------------------------------
'''
ellip_graph = False
b_ellip_init = 1.0
upsilon_init = 2.0
ellip_init = np.array([b_ellip_init, upsilon_init])
Frisch_elast = 0.8
CFE_scale = 1.0
cfe_params = np.array([Frisch_elast, CFE_scale])
b_ellip, upsilon = elp.fit_ellip_CFE(ellip_init, cfe_params, l_tilde,
                                     ellip_graph)

'''
------------------------------------------------------------------------
Solve for the steady-state solution
------------------------------------------------------------------------
cur_path       = string, current file path of this script
ss_output_fldr = string, cur_path extension of SS output folder path
ss_output_dir  = string, full path name of SS output folder
ss_outputfile  = string, path name of file for SS output objects
ss_paramsfile  = string, path name of file for SS parameter objects
Kss_init       = scalar > 0, initial guess for K_ss
Lss_init       = scalar > 0, initial guess for L_ss
rss_init       = scalar > 0, initial guess for r_ss
wss_init       = scalar > 0, initial guess for w_ss
c1_init        = scalar > 0, initial guess for c1
init_vals      = length 5 tuple, initial values to be passed in to
                 get_SS_root() or get_SS_bsct()
ss_args        = length 15 tuple, args to be passed in to get_SS()
ss_output      = length 14 dict, steady-state objects {n_ss, b_ss, c_ss,
                 b_Sp1_ss, w_ss, r_ss, K_ss, L_ss, Y_ss, C_ss, n_err_ss,
                 b_err_ss, RCerr_ss, ss_time}
ss_vars_exst   = boolean, =True if ss_vars.pkl exists
ss_args_exst   = boolean, =True if ss_args.pkl exists
err_msg        = string, error message
cur_ss_args    = length 15 tuple, current args to be passed in to
                 get_SS()
args_same      = boolean, =True if ss_args == cur_ss_args
------------------------------------------------------------------------
'''
# Create OUTPUT/SS directory if does not already exist
cur_path = os.path.split(os.path.abspath(__file__))[0]
ss_output_fldr = 'OUTPUT/SS'
ss_output_dir = os.path.join(cur_path, ss_output_fldr)
if not os.access(ss_output_dir, os.F_OK):
    os.makedirs(ss_output_dir)
ss_outputfile = os.path.join(ss_output_dir, 'ss_vars.pkl')
ss_paramsfile = os.path.join(ss_output_dir, 'ss_args.pkl')

# Compute steady-state solution
if SS_solve:
    print('BEGIN EQUILIBRIUM STEADY-STATE COMPUTATION')

    print('Solving SS outer loop using bisection method.')
    Kss_init = 200.0
    Lss_init = 100.0
    rss_init = 0.05
    wss_init = 1.2
    c1_init = 0.1
    init_vals = (Kss_init, Lss_init, rss_init, wss_init, c1_init)
    ss_args = (S, beta, sigma, l_tilde, b_ellip, upsilon, chi_n_vec,
               A, alpha, delta, tax_params, fiscal_params, SS_tol,
               SS_EulDiff, hh_fsolve_SS)
    ss_output = ss.get_SS_bsct(init_vals, ss_args, SS_graphs)

    # Save ss_output as pickle
    pickle.dump(ss_output, open(ss_outputfile, 'wb'))
    pickle.dump(ss_args, open(ss_paramsfile, 'wb'))

# Don't compute steady-state, get it from pickle
else:
    # Make sure that the SS output files exist
    ss_vars_exst = os.path.exists(ss_outputfile)
    ss_args_exst = os.path.exists(ss_paramsfile)
    if (not ss_vars_exst) or (not ss_args_exst):
        # If the files don't exist, stop the program and run the steady-
        # state solution first
        err_msg = ('ERROR: The SS output files do not exist and ' +
                   'SS_solve=False. Must set SS_solve=True and ' +
                   'compute steady-state solution.')
        raise RuntimeError(err_msg)
    else:
        # If the files do exist, make sure that none of the parameters
        # changed from the parameters used in the solution for the saved
        # steady-state pickle
        ss_args = pickle.load(open(ss_paramsfile, 'rb'))
        cur_ss_args = (S, beta, sigma, l_tilde, b_ellip, upsilon, chi_n_vec,
                   A, alpha, delta, tax_params, fiscal_params, SS_tol,
                   SS_EulDiff, hh_fsolve_SS)
        args_same = utils.compare_args(ss_args[:-1], cur_ss_args[:-1])
        if args_same:
            # If none of the parameters changed, use saved pickle
            print('RETRIEVE STEADY-STATE SOLUTIONS FROM FILE')
            ss_output = pickle.load(open(ss_outputfile, 'rb'))
        else:
            # If any of the parameters changed, end the program and
            # compute the steady-state solution
            err_msg = ('ERROR: Current ss_args are not equal to the ' +
                       'ss_args that produced ss_output. Must solve ' +
                       'for SS before solving transition path. Set ' +
                       'SS_solve=True.')
            raise RuntimeError(err_msg)

'''
------------------------------------------------------------------------
Solve for the transition path equilibrium by time path iteration (TPI)
------------------------------------------------------------------------
tpi_output_fldr = string, cur_path extension of TPI output folder path
tpi_output_dir  = string, full path name of TPI output folder
tpi_outputfile  = string, path name of file for TPI output objects
tpi_paramsfile  = string, path name of file for TPI parameter objects
B_ss            = scalar > 0, steady-state aggregate savings
K_ss            = scalar > 0, steady-state aggregate capital stock
L_ss            = scalar > 0, steady-state aggregate labor
C_ss            = scalar > 0, steady-state aggregate consumption
G_ss            = scalar, steady-state government spending
b_ss            = (S,) vector, steady-state savings distribution
init_wgts       = (S,) vector, weights representing the factor by which
                  the initial wealth distribution differs from the
                  steady-state wealth distribution
bvec1           = (S,) vector, initial period savings distribution
K1              = scalar, initial period aggregate capital stock
K1_cnstr        = Boolean, =True if K1 <= 0
tpi_params      = length 20 tuple, args to pass into c7tpf.get_TPI()
tpi_output      = length 14 dictionary, {cpath, npath, bpath, wpath,
                  rpath, Kpath, Lpath, Ypath, Cpath, bSp1_err_path,
                  b_err_path, n_err_path, RCerrPath, tpi_time}
tpi_args        = length 21 tuple, args that were passed in to get_TPI()
------------------------------------------------------------------------
'''
if TPI_solve:
    print('BEGIN EQUILIBRIUM TRANSITION PATH COMPUTATION')

    # Create OUTPUT/TPI directory if does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    tpi_output_fldr = 'OUTPUT/TPI'
    tpi_output_dir = os.path.join(cur_path, tpi_output_fldr)
    if not os.access(tpi_output_dir, os.F_OK):
        os.makedirs(tpi_output_dir)
    tpi_outputfile = os.path.join(tpi_output_dir, 'tpi_vars.pkl')
    tpi_paramsfile = os.path.join(tpi_output_dir, 'tpi_args.pkl')

    B_ss = ss_output['B_ss']
    K_ss = ss_output['K_ss']
    L_ss = ss_output['L_ss']
    C_ss = ss_output['C_ss']
    G_ss = ss_output['G_ss']
    b_splus1_ss = ss_output['b_splus1_ss']
    n_ss = ss_output['n_ss']

    # Choose initial period distribution of wealth (bvec1), which
    # determines initial period aggregate capital stock
    b_s_ss = ss_output['b_s_ss']
    init_wgts = ((1.5 - 0.87) / (S - 1) *
                 (np.linspace(1, S, S) - 1) + 0.87)
    bvec1 = init_wgts * b_s_ss
    #bvec1 = b_s_ss
    # Make sure init. period distribution is feasible in terms of K
    K1, K1_cnstr = aggr.get_K(bvec1)

    # If initial bvec1 is not feasible end program
    if K1_cnstr:
        print('Initial savings distribution is not feasible because ' +
              'K1<=0. Some element(s) of bvec1 must increase.')
    else:
        tpi_params = (S, T1, T2, beta, sigma, l_tilde, b_ellip, upsilon,
                      chi_n_vec, A, alpha, delta, tax_params, fiscal_params,
                      B_ss, K_ss, L_ss, C_ss, G_ss, b_s_ss, b_splus1_ss, n_ss, maxiter_TPI,
                      mindist_TPI, TPI_tol, xi, TPI_EulDiff, hh_fsolve_TPI)
        tpi_output = tpi.get_TPI(tpi_params, bvec1, TPI_graphs)

        tpi_args = (S, T1, T2, beta, sigma, l_tilde, b_ellip, upsilon,
                    chi_n_vec, A, alpha, delta, tax_params, fiscal_params,
                    B_ss, K_ss, L_ss, C_ss, G_ss, b_s_ss, b_splus1_ss, n_ss, maxiter_TPI,
                    mindist_TPI, TPI_tol, xi, TPI_EulDiff, hh_fsolve_TPI, bvec1)

        # Save tpi_output as pickle
        pickle.dump(tpi_output, open(tpi_outputfile, 'wb'))
        pickle.dump(tpi_args, open(tpi_paramsfile, 'wb'))
