'''
This module contains functions related to the household

This module defines the following function(s):
    get_MU_c()

'''

import numpy as np
import scipy.optimize as opt
import tax


def get_MU_c(c, sigma):
    MU_c = c ** (-sigma)

    return MU_c


def get_MDU_n(n, l_tilde, b, upsilon):
    MDU_n = ((b / l_tilde) * ((n / l_tilde) ** (upsilon - 1)) *
             (1 - ((n / l_tilde) ** upsilon)) **
             ((1 - upsilon) / upsilon))
    # n_gtl = n >= l_tilde
    # MDU_n[n_gtl] = 9999

    return MDU_n


def get_recurs_c(c1, r, beta, sigma, S):
    cvec = np.zeros(S)
    cvec[0] = c1
    c_s = c1
    for s in range(S - 1):
        c_sp1 = (((1 + r) * beta) ** (1 / sigma)) * c_s
        cvec[s + 1] = c_sp1
        c_s = c_sp1

    return cvec


def get_recurs_b(cvec, nvec, r, w):
    '''
    cvec = (S,) vector, consumption
    nvec = (S,)...

    bvec = (S,) vector, savings where each element is b_sp1
    '''
    S = cvec.shape[0]
    bvec = np.zeros(S)
    b_s = 0.0
    for s in range(S):
        b_sp1 = (1 + r) * b_s + w * nvec[s] - cvec[s]
        bvec[s] = b_sp1
        b_s = b_sp1

    return bvec


def get_n_errors(nvec, *args):
    (cvec, bvec, r, w, mtrxparams, factor, sigma, chi_n_vec, l_tilde,
        b_ellip, upsilon) = args
    MU_c = get_MU_c(cvec, sigma)
    MDU_n = get_MDU_n(nvec, l_tilde, b_ellip, upsilon)
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

    MU_csm1 = get_MU_c(c_sm1, sigma)
    MU_cs = get_MU_c(c_s, sigma)
    lab_inc = w * n_s
    cap_inc = r * b_s
    MTRy = tax.get_taxrates(lab_inc, cap_inc, factor, mtryparams)
    n_args = (c_s, b_s, r, w, mtrxparams, factor, sigma, chi_n_s, l_tilde,
              b_ellip, upsilon)
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
