'''
------------------------------------------------------------------------
This program runs the steady state solver as well as the time path
iteration solution for the model with S-period lived agents and
exogenous labor from Chapter 6 of the OG textbook.

This Python script imports the following module(s):
    Chap6ssfuncs.py
    Chap6tpfuncs.py

This Python script calls the following function(s):
    c6ssf.feasible()
    c6ssf.get_SS()
    c6ssf.get_K
    c6tpf.get_TPI()
------------------------------------------------------------------------
'''
# Import packages
import numpy as np
import Chap6ssfuncs as c6ssf
import Chap6tpfuncs as c6tpf

'''
------------------------------------------------------------------------
Declare parameters
------------------------------------------------------------------------
S            = integer in [3,80], number of periods an individual lives
beta_annual  = scalar in (0,1), discount factor for one year
beta         = scalar in (0,1), discount factor for each model period
sigma        = scalar > 0, coefficient of relative risk aversion
ncutper      = int >= 2, age at which labor supply is exogenously
               reduced
nvec         = [S,] vector, exogenous labor supply n_{s,t}
L            = scalar > 0, exogenous aggregate labor
A            = scalar > 0, total factor productivity parameter in firms'
               production function
alpha        = scalar in (0,1), capital share of income
delta_annual = scalar in [0,1], one-year depreciation rate of capital
delta        = scalar in [0,1], model-period depreciation rate of
               capital
SS_tol       = scalar > 0, tolerance level for steady-state fsolve
SS_graphs    = boolean, =True if want graphs of steady-state objects
T            = integer > S, number of time periods until steady state
TPI_solve    = boolean, =True if want to solve TPI after solving SS
TPI_tol      = scalar > 0, tolerance level for fsolve's in TPI
maxiter_TPI  = integer >= 1, Maximum number of iterations for TPI
mindist_TPI  = scalar > 0, Convergence criterion for TPI
xi           = scalar in (0,1], TPI path updating parameter
TPI_graphs   = Boolean, =True if want graphs of TPI objects
EulDiff      = Boolean, =True if want difference version of Euler errors
               beta*(1+r)*u'(c2) - u'(c1), =False if want ratio version
               [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
------------------------------------------------------------------------
'''
# Household parameters
S = int(80)
beta_annual = .96
beta = beta_annual ** (80 / S)
sigma = 3.0
ncutper = round((2 / 3) * S)
nvec = np.ones(S)
nvec[ncutper:] = 0.2
L = nvec.sum()
# Firm parameters
A = 1.0
alpha = .35
delta_annual = .05
delta = 1 - ((1 - delta_annual) ** (80 / S))
# SS parameters
SS_tol = 1e-13
SS_graphs = True
# TPI parameters
T = int(round(2.5 * S))
TPI_solve = True
TPI_tol = 1e-13
maxiter_TPI = 300
mindist_TPI = 1e-13
xi = 0.20
TPI_graphs = True
# Overall parameters
EulDiff = False

# '''
# ------------------------------------------------------------------------
# Check feasibility
# ------------------------------------------------------------------------
# f_params    = length 4 tuple, (nvec, A, alpha, delta)
# bvec_guess1 = (S-1,) vector, guess for steady-state bvec (b2,b3,...b_S)
# b_cnstr     = (S-1,) Boolean vector, =True if b_s causes negative
#               consumption c_s <= 0 or negative aggregate capital stock
#               K <= 0
# c_cnstr     = (S,) Boolean vector, =True for elements of negative
#               consumption c_s <= 0
# K_cnstr     = Boolean, =True if K <= 0
# bvec_guess2 = (S-1,) vector, guess for steady-state bvec (b2,b3,...b_S)
# bvec_guess3 = (S-1,) vector, guess for steady-state bvec (b2,b3,...b_S)

# ------------------------------------------------------------------------
# '''
# f_params = (nvec, A, alpha, delta)

# bvec_guess1 = np.ones(S - 1)
# b_cnstr, c_cnstr, K_cnstr = c6ssf.feasible(f_params, bvec_guess1)
# print('bvec_guess1', bvec_guess1)
# print('c_cnstr', c_cnstr)
# print('b_cnstr', b_cnstr)
# print('K_cnstr', K_cnstr)

# bvec_guess2 = np.array([-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
#                         -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
#                         -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
#                         -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
#                         -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
#                         -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
#                         -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
#                         -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
#                         -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
#                         -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2])
# b_cnstr, c_cnstr, K_cnstr = c6ssf.feasible(f_params, bvec_guess2)
# print('bvec_guess2', bvec_guess2)
# print('c_cnstr', c_cnstr)
# print('b_cnstr', b_cnstr)
# print('K_cnstr', K_cnstr)

# bvec_guess3 = np.array([-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
#                         -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
#                         -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
#                         -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
#                         -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
#                         -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
#                         -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
#                         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
#                         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
#                         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
# b_cnstr, c_cnstr, K_cnstr = c6ssf.feasible(f_params, bvec_guess3)
# print('bvec_guess3', bvec_guess3)
# print('c_cnstr', c_cnstr)
# print('b_cnstr', b_cnstr)
# print('K_cnstr', K_cnstr)

# bvec_guess4 = 0.1 * np.ones(S - 1)
# b_cnstr, c_cnstr, K_cnstr = c6ssf.feasible(f_params, bvec_guess4)
# print('bvec_guess3', bvec_guess4)
# print('c_cnstr', c_cnstr)
# print('b_cnstr', b_cnstr)
# print('K_cnstr', K_cnstr)

'''
------------------------------------------------------------------------
Run the steady-state solution
------------------------------------------------------------------------
bvec_guess = (S-1,) vector, initial guess for steady-state bvec
             (b2, b3, ...b_S)
f_params   = length 4 tuple, (nvec, A, alpha, delta)
b_cnstr    = (S-1,) Boolean vector, =True if b_s causes negative
             consumption c_s <= 0 or negative aggregate capital stock
             K <= 0
c_cnstr    = (S,) Boolean vector, =True for elements of negative
             consumption c_s <= 0
K_cnstr    = Boolean, =True if K <= 0
ss_params  = length 9 tuple,
             (beta, sigma, nvec, L, A, alpha, delta, SS_tol, EulDiff)
ss_output  = length 10 dictionary, {b_ss, c_ss, w_ss, r_ss, K_ss, Y_ss,
             C_ss, EulErr_ss, RCerr_ss, ss_time}
------------------------------------------------------------------------
'''
print('BEGIN EQUILIBRIUM STEADY-STATE COMPUTATION')
bvec_guess = np.array([-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
                        -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
                        -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
                        -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
                        -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
                        -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
                        -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
                        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
f_params = (nvec, A, alpha, delta)
b_cnstr, c_cnstr, K_cnstr = c6ssf.feasible(f_params, bvec_guess)
if not K_cnstr and not c_cnstr.max():
    ss_params = (beta, sigma, nvec, L, A, alpha, delta, SS_tol,
                 EulDiff)
    ss_output = c6ssf.get_SS(ss_params, bvec_guess, SS_graphs)
else:
    print("Initial guess for SS bvec does not satisfy K>0 or c_s>0.")

'''
------------------------------------------------------------------------
Run the time path iteration (TPI) solution
------------------------------------------------------------------------
b_ss         = (S-1,) vector, steady-state savings distribution
K_ss         = scalar > 0, steady-state aggregate capital stock
C_ss         = scalar > 0, steady-state aggregate consumption
bvec1        = (S-1,) vector, initial period savings distribution
K1           = scalar, initial period aggregate capital stock
K_constr_tp1 = Boolean, =True if K1 <= 0
tpi_params   = length 17 tuple, (S, T, beta, sigma, nvec, L, A, alpha,
               delta, b_ss, K_ss, C_ss, maxiter_TPI, mindist_TPI, xi,
               TPI_tol, EulDiff)
tpi_output   = length 10 dictionary, {bpath, cpath, wpath, rpath, Kpath,
               Ypath, Cpath, EulErrPath, RCerrPath, tpi_time}
------------------------------------------------------------------------
'''
if TPI_solve:
    print('BEGIN EQUILIBRIUM TIME PATH COMPUTATION')
    b_ss = ss_output['b_ss']
    K_ss = ss_output['K_ss']
    C_ss = ss_output['C_ss']
    init_wgts = ((1.5 - 0.87) / (S - 2) *
                 (np.linspace(2, S, S - 1) - 2) + 0.87)
    bvec1 = init_wgts * b_ss

    # Make sure init. period distribution is feasible in terms of K
    K1, K_constr_tpi1 = c6ssf.get_K(bvec1)
    if K_constr_tpi1:
        print('Initial savings distribution is not feasible because ' +
              'K1<=0. Some element(s) of bvec1 must increase.')
    else:
        tpi_params = (S, T, beta, sigma, nvec, L, A, alpha, delta, b_ss,
                      K_ss, C_ss, maxiter_TPI, mindist_TPI, xi, TPI_tol,
                      EulDiff)
        tpi_output = c6tpf.get_TPI(tpi_params, bvec1, TPI_graphs)
