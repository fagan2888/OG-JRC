'''
This module contains functions related to the household

This module defines the following function(s):
    get_MU_c()

'''

import numpy as np
import scipy.optimize as opt
import tax
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os


# def get_MU_c(c, sigma):
#     MU_c = c ** (-sigma)

#     return MU_c


# def get_MDU_n(n, l_tilde, b, upsilon):
#     MDU_n = ((b / l_tilde) * ((n / l_tilde) ** (upsilon - 1)) *
#              (1 - ((n / l_tilde) ** upsilon)) **
#              ((1 - upsilon) / upsilon))
#     # n_gtl = n >= l_tilde
#     # MDU_n[n_gtl] = 9999

#     return MDU_n


def MU_c_stitch(cvec, sigma, graph=False):
    '''
    --------------------------------------------------------------------
    Generate marginal utility(ies) of consumption with CRRA consumption
    utility and stitched function at lower bound such that the new
    hybrid function is defined over all consumption on the real
    line but the function has similar properties to the Inada condition.

    u'(c) = c ** (-sigma) if c >= epsilon
          = g'(c) = 2 * b2 * c + b1 if c < epsilon

        such that g'(epsilon) = u'(epsilon)
        and g''(epsilon) = u''(epsilon)

        u(c) = (c ** (1 - sigma) - 1) / (1 - sigma)
        g(c) = b2 * (c ** 2) + b1 * c + b0
    --------------------------------------------------------------------
    INPUTS:
    cvec  = scalar or (p,) vector, individual consumption value or
            lifetime consumption over p consecutive periods
    sigma = scalar >= 1, coefficient of relative risk aversion for CRRA
            utility function: (c**(1-sigma) - 1) / (1 - sigma)
    graph = boolean, =True if want plot of stitched marginal utility of
            consumption function

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    epsilon    = scalar > 0, positive value close to zero
    c_s        = scalar, individual consumption
    c_s_cnstr  = boolean, =True if c_s < epsilon
    b1         = scalar, intercept value in linear marginal utility
    b2         = scalar, slope coefficient in linear marginal utility
    MU_c       = scalar or (p,) vector, marginal utility of consumption
                 or vector of marginal utilities of consumption
    p          = integer >= 1, number of periods remaining in lifetime
    cvec_cnstr = (p,) boolean vector, =True for values of cvec < epsilon

    FILES CREATED BY THIS FUNCTION:
        MU_c_stitched.png

    RETURNS: MU_c
    --------------------------------------------------------------------
    '''
    epsilon = 0.0001
    if np.ndim(cvec) == 0:
        c_s = cvec
        c_s_cnstr = c_s < epsilon
        if c_s_cnstr:
            b2 = (-sigma * (epsilon ** (-sigma - 1))) / 2
            b1 = (epsilon ** (-sigma)) - 2 * b2 * epsilon
            MU_c = 2 * b2 * c_s + b1
        else:
            MU_c = c_s ** (-sigma)
    elif np.ndim(cvec) == 1:
        p = cvec.shape[0]
        cvec_cnstr = cvec < epsilon
        MU_c = np.zeros(p)
        MU_c[~cvec_cnstr] = cvec[~cvec_cnstr] ** (-sigma)
        b2 = (-sigma * (epsilon ** (-sigma - 1))) / 2
        b1 = (epsilon ** (-sigma)) - 2 * b2 * epsilon
        MU_c[cvec_cnstr] = 2 * b2 * cvec[cvec_cnstr] + b1

    if graph:
        '''
        ----------------------------------------------------------------
        cur_path    = string, path name of current directory
        output_fldr = string, folder in current path to save files
        output_dir  = string, total path of images folder
        output_path = string, path of file name of figure to be saved
        cvec_CRRA   = (1000,) vector, support of c including values
                      between 0 and epsilon
        MU_CRRA     = (1000,) vector, CRRA marginal utility of
                      consumption
        cvec_stitch = (500,) vector, stitched support of consumption
                      including negative values up to epsilon
        MU_stitch   = (500,) vector, stitched marginal utility of
                      consumption
        ----------------------------------------------------------------
        '''
        # Create directory if images directory does not already exist
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        # Plot steady-state consumption and savings distributions
        cvec_CRRA = np.linspace(epsilon / 2, epsilon * 3, 1000)
        MU_CRRA = cvec_CRRA ** (-sigma)
        cvec_stitch = np.linspace(-0.00005, epsilon, 500)
        MU_stitch = 2 * b2 * cvec_stitch + b1
        fig, ax = plt.subplots()
        plt.plot(cvec_CRRA, MU_CRRA, ls='solid', label='$u\'(c)$: CRRA')
        plt.plot(cvec_stitch, MU_stitch, ls='dashed', color='red',
                 label='$g\'(c)$: stitched')
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Marginal utility of consumption with stitched ' +
                  'function', fontsize=14)
        plt.xlabel(r'Consumption $c$')
        plt.ylabel(r'Marginal utility $u\'(c)$')
        plt.xlim((-0.00005, epsilon * 3))
        # plt.ylim((-1.0, 1.15 * (b_ss.max())))
        plt.legend(loc='upper right')
        output_path = os.path.join(output_dir, "MU_c_stitched")
        plt.savefig(output_path)
        # plt.show()

    return MU_c


def MDU_n_stitch(nvec, params, graph=False):
    '''
    --------------------------------------------------------------------
    Generate marginal disutility(ies) of labor with elliptical
    disutility of labor function and stitched functions at lower bound
    and upper bound of labor supply such that the new hybrid function is
    defined over all labor supply on the real line but the function has
    similar properties to the Inada conditions at the upper and lower
    bounds.

    v'(n) = (b / l_tilde) * ((n / l_tilde) ** (upsilon - 1)) *
            ((1 - ((n / l_tilde) ** upsilon)) ** ((1-upsilon)/upsilon))
            if n >= eps_low <= n <= eps_high
          = g_low'(n)  = 2 * b2 * n + b1 if n < eps_low
          = g_high'(n) = 2 * d2 * n + d1 if n > eps_high

        such that g_low'(eps_low) = u'(eps_low)
        and g_low''(eps_low) = u''(eps_low)
        and g_high'(eps_high) = u'(eps_high)
        and g_high''(eps_high) = u''(eps_high)

        v(n) = -b *(1 - ((n/l_tilde) ** upsilon)) ** (1/upsilon)
        g_low(n)  = b2 * (n ** 2) + b1 * n + b0
        g_high(n) = d2 * (n ** 2) + d1 * n + d0
    --------------------------------------------------------------------
    INPUTS:
    nvec   = scalar or (p,) vector, labor supply value or labor supply
             values over remaining periods of lifetime
    params = length 3 tuple, (l_tilde, b_ellip, upsilon)
    graph  = Boolean, =True if want plot of stitched marginal disutility
             of labor function

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    l_tilde       = scalar > 0, time endowment for each agent each per
    b_ellip       = scalar > 0, scale parameter for elliptical utility
                    of leisure function
    upsilon       = scalar > 1, shape parameter for elliptical utility
                    of leisure function
    eps_low       = scalar > 0, positive value close to zero
    eps_high      = scalar > 0, positive value just less than l_tilde
    n_s           = scalar, individual labor supply
    n_s_low       = boolean, =True for n_s < eps_low
    n_s_high      = boolean, =True for n_s > eps_high
    n_s_uncstr    = boolean, =True for n_s >= eps_low and
                    n_s <= eps_high
    MDU_n         = scalar or (p,) vector, marginal disutility or
                    marginal utilities of labor supply
    b1            = scalar, intercept value in linear marginal
                    disutility of labor at lower bound
    b2            = scalar, slope coefficient in linear marginal
                    disutility of labor at lower bound
    d1            = scalar, intercept value in linear marginal
                    disutility of labor at upper bound
    d2            = scalar, slope coefficient in linear marginal
                    disutility of labor at upper bound
    p             = integer >= 1, number of periods remaining in life
    nvec_s_low    = boolean, =True for n_s < eps_low
    nvec_s_high   = boolean, =True for n_s > eps_high
    nvec_s_uncstr = boolean, =True for n_s >= eps_low and
                    n_s <= eps_high

    FILES CREATED BY THIS FUNCTION:
        MDU_n_stitched.png

    RETURNS: MDU_n
    --------------------------------------------------------------------
    '''
    l_tilde, b_ellip, upsilon = params
    eps_low = 0.000001
    eps_high = l_tilde - 0.000001
    # This if is for when nvec is a scalar
    if np.ndim(nvec) == 0:
        n_s = nvec
        n_s_low = n_s < eps_low
        n_s_high = n_s > eps_high
        n_s_uncstr = (n_s >= eps_low) and (n_s <= eps_high)
        if n_s_uncstr:
            MDU_n = \
                ((b_ellip / l_tilde) * ((n_s / l_tilde) **
                 (upsilon - 1)) * ((1 - ((n_s / l_tilde) ** upsilon)) **
                 ((1 - upsilon) / upsilon)))
        elif n_s_low:
            b2 = (0.5 * b_ellip * (l_tilde ** (-upsilon)) *
                  (upsilon - 1) * (eps_low ** (upsilon - 2)) *
                  ((1 - ((eps_low / l_tilde) ** upsilon)) **
                  ((1 - upsilon) / upsilon)) *
                  (1 + ((eps_low / l_tilde) ** upsilon) *
                  ((1 - ((eps_low / l_tilde) ** upsilon)) ** (-1))))
            b1 = ((b_ellip / l_tilde) * ((eps_low / l_tilde) **
                  (upsilon - 1)) *
                  ((1 - ((eps_low / l_tilde) ** upsilon)) **
                  ((1 - upsilon) / upsilon)) - (2 * b2 * eps_low))
            MDU_n = 2 * b2 * n_s + b1
        elif n_s_high:
            d2 = (0.5 * b_ellip * (l_tilde ** (-upsilon)) *
                  (upsilon - 1) * (eps_high ** (upsilon - 2)) *
                  ((1 - ((eps_high / l_tilde) ** upsilon)) **
                  ((1 - upsilon) / upsilon)) *
                  (1 + ((eps_high / l_tilde) ** upsilon) *
                  ((1 - ((eps_high / l_tilde) ** upsilon)) ** (-1))))
            d1 = ((b_ellip / l_tilde) * ((eps_high / l_tilde) **
                  (upsilon - 1)) *
                  ((1 - ((eps_high / l_tilde) ** upsilon)) **
                  ((1 - upsilon) / upsilon)) - (2 * d2 * eps_high))
            MDU_n = 2 * d2 * n_s + d1
    # This if is for when nvec is a one-dimensional vector
    elif np.ndim(nvec) == 1:
        p = nvec.shape[0]
        nvec_low = nvec < eps_low
        nvec_high = nvec > eps_high
        nvec_uncstr = np.logical_and(~nvec_low, ~nvec_high)
        MDU_n = np.zeros(p)
        MDU_n[nvec_uncstr] = (
            (b_ellip / l_tilde) *
            ((nvec[nvec_uncstr] / l_tilde) ** (upsilon - 1)) *
            ((1 - ((nvec[nvec_uncstr] / l_tilde) ** upsilon)) **
             ((1 - upsilon) / upsilon)))
        b2 = (0.5 * b_ellip * (l_tilde ** (-upsilon)) * (upsilon - 1) *
              (eps_low ** (upsilon - 2)) *
              ((1 - ((eps_low / l_tilde) ** upsilon)) **
              ((1 - upsilon) / upsilon)) *
              (1 + ((eps_low / l_tilde) ** upsilon) *
              ((1 - ((eps_low / l_tilde) ** upsilon)) ** (-1))))
        b1 = ((b_ellip / l_tilde) * ((eps_low / l_tilde) **
              (upsilon - 1)) *
              ((1 - ((eps_low / l_tilde) ** upsilon)) **
              ((1 - upsilon) / upsilon)) - (2 * b2 * eps_low))
        MDU_n[nvec_low] = 2 * b2 * nvec[nvec_low] + b1
        d2 = (0.5 * b_ellip * (l_tilde ** (-upsilon)) * (upsilon - 1) *
              (eps_high ** (upsilon - 2)) *
              ((1 - ((eps_high / l_tilde) ** upsilon)) **
              ((1 - upsilon) / upsilon)) *
              (1 + ((eps_high / l_tilde) ** upsilon) *
              ((1 - ((eps_high / l_tilde) ** upsilon)) ** (-1))))
        d1 = ((b_ellip / l_tilde) * ((eps_high / l_tilde) **
              (upsilon - 1)) *
              ((1 - ((eps_high / l_tilde) ** upsilon)) **
              ((1 - upsilon) / upsilon)) - (2 * d2 * eps_high))
        MDU_n[nvec_high] = 2 * d2 * nvec[nvec_high] + d1

    if graph:
        '''
        ----------------------------------------------------------------
        cur_path    = string, path name of current directory
        output_fldr = string, folder in current path to save files
        output_dir  = string, total path of images folder
        output_path = string, path of file name of figure to be saved
        nvec_ellip  = (1000,) vector, support of n including values
                      between 0 and eps_low and between eps_high and
                      l_tilde
        MU_CRRA     = (1000,) vector, CRRA marginal utility of
                      consumption
        cvec_stitch = (500,) vector, stitched support of consumption
                      including negative values up to epsilon
        MU_stitch   = (500,) vector, stitched marginal utility of
                      consumption
        ----------------------------------------------------------------
        '''
        # Create directory if images directory does not already exist
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        # Plot steady-state consumption and savings distributions
        nvec_ellip = np.linspace(eps_low / 2, eps_high +
                                 ((l_tilde - eps_high) / 5), 1000)
        MDU_ellip = (
            (b_ellip / l_tilde) *
            ((nvec_ellip / l_tilde) ** (upsilon - 1)) *
            ((1 - ((nvec_ellip / l_tilde) ** upsilon)) **
             ((1 - upsilon) / upsilon)))
        n_stitch_low = np.linspace(-0.05, eps_low, 500)
        MDU_stitch_low = 2 * b2 * n_stitch_low + b1
        n_stitch_high = np.linspace(eps_high, l_tilde + 0.000005, 500)
        MDU_stitch_high = 2 * d2 * n_stitch_high + d1
        fig, ax = plt.subplots()
        plt.plot(nvec_ellip, MDU_ellip, ls='solid', color='black',
                 label='$v\'(n)$: Elliptical')
        plt.plot(n_stitch_low, MDU_stitch_low, ls='dashed', color='red',
                 label='$g\'(n)$: low stitched')
        plt.plot(n_stitch_high, MDU_stitch_high, ls='dotted',
                 color='blue', label='$g\'(n)$: high stitched')
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Marginal utility of consumption with stitched ' +
                  'function', fontsize=14)
        plt.xlabel(r'Labor $n$')
        plt.ylabel(r'Marginal disutility $v\'(n)$')
        plt.xlim((-0.05, l_tilde + 0.01))
        # plt.ylim((-1.0, 1.15 * (b_ss.max())))
        plt.legend(loc='upper left')
        output_path = os.path.join(output_dir, "MDU_n_stitched")
        plt.savefig(output_path)
        # plt.show()

    return MDU_n


def get_n_errors(nvec, *args):
    (cvec, bvec, r, w, mtrxparams, factor, sigma, chi_n_vec, l_tilde,
        b_ellip, upsilon) = args
    MU_c = MU_c_stitch(cvec, sigma)
    MDU_n = MDU_n_stitch(nvec, (l_tilde, b_ellip, upsilon))
    lab_inc = w * nvec
    cap_inc = r * bvec
    MTRx = tax.get_taxrates(lab_inc, cap_inc, factor, mtrxparams)
    n_errors = w * (1 - MTRx) * MU_c - chi_n_vec * MDU_n

    return n_errors


def get_n_s(n_args):
    (cvec, bvec, r, w, mtrxparams, factor, sigma, chi_n_vec, l_tilde,
        b_ellip, upsilon) = n_args
    n_guess = 0.5 * l_tilde * np.ones_like(cvec)
    results_n = opt.root(get_n_errors, n_guess, args=(n_args),
                         method='lm')
    nvec = results_n.x

    return nvec


def get_savings(c_s, n_s, b_s, args):
    (r, w, X, factor, etrparams) = args
    lab_inc = w * n_s
    cap_inc = r * b_s
    tot_tax_liab = tax.get_tot_tax_liab(lab_inc, cap_inc, factor,
                                        etrparams)
    b_sp1 = (1 + r) * b_s + w * n_s + X - tot_tax_liab - c_s

    return b_sp1


def get_cn(cn_vec, *args):
    '''
    EulErrors = (2,) vector, errors from 2 household Euler equations
    '''
    (c_sm1, b_s, r, w, factor, beta, sigma, chi_n_s, l_tilde, b_ellip,
        upsilon, mtrxparams, mtryparams) = args
    c_s, n_s = cn_vec

    MU_csm1 = MU_c_stitch(c_sm1, sigma)
    MU_cs = MU_c_stitch(c_s, sigma)
    lab_inc = w * n_s
    cap_inc = r * b_s
    MTRy = tax.get_taxrates(lab_inc, cap_inc, factor, mtryparams)
    n_args = (c_s, b_s, r, w, mtrxparams, factor, sigma, chi_n_s,
              l_tilde, b_ellip, upsilon)
    # print(c_s, n_s)
    error1 = get_n_errors(n_s, *n_args)
    error2 = MU_csm1 - beta * (1 + r * (1 - MTRy)) * MU_cs
    EulErrors = np.array([error1, error2])

    return EulErrors


def get_cnbvecs(c1, args):
    '''
    bvec = (S,) vector, b_2, b3,...b_{S+1}
    '''
    (r, w, X, factor, beta, sigma, chi_n_vec, l_tilde, b_ellip, upsilon,
        S, etrparams, mtrxparams, mtryparams) = args
    cvec = np.zeros(S)
    cvec[0] = c1
    nvec = np.zeros(S)
    bvec = np.zeros(S)

    n1_args = (c1, 0.0, r, w, mtrxparams, factor, sigma, chi_n_vec[0],
               l_tilde, b_ellip, upsilon)
    n1 = get_n_s(n1_args)
    nvec[0] = n1
    bs_args = (r, w, X, factor, etrparams)
    b2 = get_savings(c1, n1, 0.0, bs_args)
    bvec[0] = b2
    for s in range(1, S):
        cs_init = cvec[s - 1]
        ns_init = nvec[s - 1]
        cn_init = np.array([cs_init, ns_init])
        c_sm1 = cvec[s - 1]
        b_s = bvec[s - 1]
        chi_n_s = chi_n_vec[s]
        cn_args = (c_sm1, b_s, r, w, factor, beta, sigma, chi_n_s,
                   l_tilde, b_ellip, upsilon, mtrxparams, mtryparams)
        results_cn = opt.root(get_cn, cn_init, args=(cn_args))
        c_s, n_s = results_cn.x
        cvec[s] = c_s
        nvec[s] = n_s
        b_s = bvec[s - 1]
        b_sp1 = get_savings(c_s, n_s, b_s, bs_args)
        bvec[s] = b_sp1

    return cvec, nvec, bvec


def get_bSp1(c1, *args):
    cvec, nvec, bvec = get_cnbvecs(c1, args)
    b_Sp1 = bvec[-1]

    return b_Sp1
