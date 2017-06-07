'''
------------------------------------------------------------------------
This module contains the functions used to solve for the values of
fiscal variables, namely government spending and debt. This module
is used to solve for the model with S-period lived agents and endogenous
labor supply and an unbalanced government budget constraint from Chapter
15 of the OG textbook.

This Python module imports the following module(s):

This Python module defines the following function(s):
    D_G_path()

------------------------------------------------------------------------
'''

# Packages
import numpy as np

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''



def D_G_path(dg_fixed_values, fiscal_params, dg_params):
    '''
    --------------------------------------------------------------------
    Solves for government spending and debt over the transition path
    --------------------------------------------------------------------
    INPUTS:
    dg_fixed_values = length 5 tuple, (Ypath, Rpath, agg_Xpath,
                                       D_0, G_0)
    fiscal_params   = length 7 tuple, (tG1, tG2, alpha_X, alpha_G, rho_G,
                                       alpha_D, alpha_D0)
    dg_params     = length 2 tuple, (T2, rpath)


    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:


    OBJECTS CREATED WITHIN FUNCTION:
    tG1            = integer, model period when budget closure rule begins
    tG2            = integer, model period when budget is closed
    alpha_X        = scalar, ratio of lump sum transfers to GDP
    alpha_G        = scalar, ratio of government spending to GDP prior to
                            budget closure rule beginning
    rho_G          = scalar in (0,1), rate of convergence to SS budget
    alpha_D  = scalar, steady-state debt to GDP ratio
    alpha_D0   = scalar, debt to GDP ratio in the initial period
    T2             = integer > T1, number of time periods after which
                     steady-state is forced in TPI
    rpath          = (T2+S-1,) vector, time path of the interest rates
    Ypath          = (T2+S-1,) vector, equilibrium time path of aggregate
                    output (GDP) Y_t
    Rpath   = (T2+S-1,) vector, path of total tax revenues
    Xpath = (T2+S-1,) vector, path of aggregate lump sum transfers
    D0             = scalar, initial period debt level
    G0             = scalar, initial period amount of government spending
    Dpath          = (T2+S-1,) vector, path of government debt
    Gpath          = (T2+S-1,) vector, path of government spending
    t              = integer, model period along transition path
    D_ratio_max    = scalar, maximum debt to GDP ratio along transition path


    FILES CREATED BY THIS FUNCTION:

    RETURNS: Dpath, Gpath
    --------------------------------------------------------------------
    '''
    tG1, tG2, alpha_X, alpha_G, rho_G, alpha_D, alpha_D0 = fiscal_params

    T2, rpath = dg_params
    Ypath, Rpath, Xpath, D0, G0 = dg_fixed_values

    Dpath = np.zeros(T2+1)
    Dpath[0] = D0
    Gpath = alpha_G * Ypath[:T2]
    Gpath[0] = G0

    t = 1
    while t < T2-1:
        Dpath[t] = ((1+rpath[t-1])*Dpath[t-1] + Gpath[t-1] +
                    Xpath[t-1] - Rpath[t-1])

        #debt_service = r[t]*D[t]
        if (t >= tG1) and (t < tG2):
            Gpath[t] = ((rho_G*alpha_D*Ypath[t] + (1-rho_G)*Dpath[t]) -
                       (1+rpath[t])*Dpath[t] + Rpath[t] -
                       Xpath[t])
        elif t >= tG2:
            Gpath[t] = ((alpha_D*Ypath[t]) - (1+rpath[t])*Dpath[t] +
                       Rpath[t] - Xpath[t])
        t += 1

    # in final period
    t = T2-1
    Dpath[t] = ((1+rpath[t-1])*Dpath[t-1] + Gpath[t-1] + Xpath[t-1]
                - Rpath[t-1])
    #debt_service = r_gov[t]*D[t]
    Gpath[t] = ((alpha_D*Ypath[t]) - (1+rpath[t])*Dpath[t] +
                Rpath[t] - Xpath[t])
    Dpath[t+1] = ((1+rpath[t])*Dpath[t] + Gpath[t] + Xpath[t] -
                  Rpath[t])
    D_ratio_max = np.amax(Dpath[:T2] / Ypath[:T2])
    print('Maximum debt ratio: ', D_ratio_max)

    return Dpath[:T2], Gpath[:T2]
