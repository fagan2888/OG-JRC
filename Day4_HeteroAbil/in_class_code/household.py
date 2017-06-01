'''
This module contains functions related to the household

This module defines the following function(s):
    get_MU_c()

'''

import numpy as np
import scipy.optimize as opt


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


def get_recurs_b(cvec, nvec, r, w, evec):
    '''
    cvec = (S,) vector, consumption
    nvec = (S,)...

    bvec = (S,) vector, savings where each element is b_sp1
    '''
    S = cvec.shape[0]
    bvec = np.zeros(S)
    b_s = 0.0
    for s in range(S):
        b_sp1 = (1 + r) * b_s + w * evec[s] * nvec[s] - cvec[s]
        bvec[s] = b_sp1
        b_s = b_sp1

    return bvec


def get_n_errors(nvec, *args):
    cvec, w, sigma, chi_n_vec, l_tilde, b, upsilon, evec = args
    MU_c = get_MU_c(cvec, sigma)
    MDU_n = get_MDU_n(nvec, l_tilde, b, upsilon)
    n_errors = w * evec * MU_c - chi_n_vec * MDU_n

    return n_errors


def get_n_s(cvec, w, sigma, chi_n_vec, l_tilde, b, upsilon, evec):
    n_args = (cvec, w, sigma, chi_n_vec, l_tilde, b, upsilon, evec)
    n_guess = 0.5 * l_tilde * np.ones_like(cvec)
    results_n = opt.root(get_n_errors, n_guess, args=(n_args),
                         method='lm')
    nvec = results_n.x

    return nvec


def get_bSp1(c1, *args):
    r, w, beta, sigma, chi_n_vec, l_tilde, b, upsilon, evec, S = args
    cvec = get_recurs_c(c1, r, beta, sigma, S)
    nvec = get_n_s(cvec, w, sigma, chi_n_vec, l_tilde, b, upsilon, evec)
    bvec = get_recurs_b(cvec, nvec, r, w, evec)
    b_Sp1 = bvec[-1]
    # print('c1: ', c1, ', b_Sp1: ', b_Sp1)

    return b_Sp1
