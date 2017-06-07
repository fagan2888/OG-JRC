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


def get_c_errors(c_s, *args):
    (c_sm1, n_s, b_s, r, w, factor, beta, sigma, mtryparam_vec) = args
    lab_inc = w * n_s
    cap_inc = r * b_s
    mtr_y = tax.get_taxrates(lab_inc, cap_inc, factor, mtryparam_vec)
    MU_c = get_MU_c(c_s, sigma)
    MU_cm1 = get_MU_c(c_sm1, sigma)
    error = MU_cm1 - beta * (1 + (1 - mtr_y) * r) * MU_c

    return error


def get_savings(c_s, n_s, b_s, r, w, X, factor, etrparam_vec):
    lab_inc = w * n_s
    cap_inc = r * b_s
    total_tax_liab = tax.get_tot_tax_liab(lab_inc, cap_inc, factor, etrparam_vec)
    b_sp1 = (1 + r) * b_s + w * n_s + X - c_s - total_tax_liab

    return b_sp1


def get_n_errors(n_s, *args):
    c_s, b_s, r, w, factor, sigma, chi_n_vec, l_tilde, b_ellip, upsilon, mtrxparam_vec = args
    lab_inc = w * n_s
    cap_inc = r * b_s
    MTRx = tax.get_taxrates(lab_inc, cap_inc, factor, mtrxparam_vec)
    MU_c = get_MU_c(c_s, sigma)
    MDU_n = get_MDU_n(n_s, l_tilde, b_ellip, upsilon)
    n_errors = (1-MTRx) * w * MU_c - chi_n_vec * MDU_n

    return n_errors


def get_n_s(c_s, b_s, r, w, factor, params):
    sigma, chi_n_vec, l_tilde, b_ellip, upsilon, mtrxparam_vec = params
    n_args = (c_s, b_s, r, w, factor, sigma, chi_n_vec, l_tilde, b_ellip, upsilon, mtrxparam_vec)
    n_guess = 0.5 * l_tilde
    results_n = opt.root(get_n_errors, n_guess, args=(n_args), method='lm')
    n_s = results_n.x

    return n_s


def get_cn(guesses, *args):
    c_s, n_s = guesses
    (c_sm1, b_s, r, w, factor, beta, sigma, chi_n_vec, l_tilde, b_ellip, upsilon, mtrxparam_vec, mtryparam_vec) = args
    n_args = (c_s, b_s, r, w, factor, sigma, chi_n_vec, l_tilde, b_ellip, upsilon, mtrxparam_vec)
    error1 = get_n_errors(n_s, *n_args)
    c_args = (c_sm1, n_s, b_s, r, w, factor, beta, sigma, mtryparam_vec)
    error2 = get_c_errors(c_s, *c_args)
    
    errors = np.array([error1, error2])

    return errors


def get_cnbvecs(c1, args):
    '''
    bvec = (S,) vector, b_2, b_3, ..., b_Sp1
    '''
    (r, w, X, factor, beta, sigma, chi_n_vec, l_tilde, b_ellip, upsilon, S, etrparam_vec, mtrxparam_vec, mtryparam_vec) = args
    cvec = np.zeros(S)
    nvec= np.zeros(S)
    bvec = np.zeros(S)
    cvec[0] = c1
    n_params = (sigma, chi_n_vec[0], l_tilde, b_ellip, upsilon, mtrxparam_vec)
    nvec[0] = get_n_s(cvec[0], 0.0, r, w, factor, n_params)
    bvec[0] = get_savings(cvec[0], nvec[0], 0.0, r, w, X, factor,
                          etrparam_vec)
    for s in range(1, S):
        cn_init = np.array([cvec[s-1], nvec[s-1]])
        cn_args = (cvec[s-1], bvec[s-1], r, w, factor, beta, sigma, chi_n_vec[s], l_tilde, b_ellip,
                   upsilon, mtrxparam_vec, mtryparam_vec)
        results_cn = opt.root(get_cn, cn_init, args=(cn_args), method='lm')
        cvec[s], nvec[s] = results_cn.x
        bvec[s] = get_savings(cvec[s], nvec[s], bvec[s-1], r, w, X, factor,
                              etrparam_vec)

    return cvec, nvec, bvec


def get_bSp1(c1, *args):
    cvec, nvec, bvec = get_cnbvecs(c1, args)
    b_Sp1 = bvec[-1]
    # print('c1: ', c1, ', b_Sp1: ', b_Sp1)

    return b_Sp1
