'''
This module contains functions related to the steady-state

This module defines the following function(s):
    get_SS()

'''

import numpy as np
import scipy.optimize as opt
import os
import household as hh
import firms
import aggregates as aggr
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def get_SS(args, graph=False):
    (KL_init, beta, sigma, chi_n_vec, l_tilde, b, upsilon, S, alpha, A,
        delta) = args
    dist = 10
    mindist = 1e-08
    maxiter = 500
    ss_iter = 0
    xi = 0.2

    r_params = (alpha, A, delta)
    w_params = (alpha, A)

    while dist > mindist and ss_iter < maxiter:
        ss_iter += 1
        K, L = KL_init
        r = firms.get_r(K, L, r_params)
        w = firms.get_w(K, L, w_params)
        c1_guess = 1.0
        c1_args = (r, w, beta, sigma, chi_n_vec, l_tilde, b, upsilon, S)
        results_c1 = opt.root(hh.get_bSp1, c1_guess, args=(c1_args))
        c1 = results_c1.x
        cvec = hh.get_recurs_c(c1, r, beta, sigma, S)
        nvec = hh.get_n_s(cvec, w, sigma, chi_n_vec, l_tilde, b,
                          upsilon)
        bvec = hh.get_recurs_b(cvec, nvec, r, w)
        K_new = aggr.get_K(bvec[:-1])
        L_new = aggr.get_L(nvec)
        KL_new = np.array([K_new, L_new])
        dist = ((KL_new - KL_init) ** 2).sum()
        KL_init = xi * KL_new + (1 - xi) * KL_init
        print('iter:', ss_iter, ' dist: ', dist)

    c_ss = cvec
    n_ss = nvec
    b_ss = bvec
    K_ss = K_new
    L_ss = L_new
    r_ss = r
    w_ss = w
    Y_params = (alpha, A)
    Y_ss = aggr.get_Y(K_ss, L_ss, Y_params)
    C_ss = aggr.get_C(c_ss)

    ss_output = {'c_ss': c_ss, 'n_ss': n_ss, 'b_ss': b_ss, 'K_ss': K_ss,
                 'L_ss': L_ss, 'r_ss': r_ss, 'w_ss': w_ss, 'Y_ss': Y_ss,
                 'C_ss': C_ss}

    if graph:
        # Create directory if images directory does not already exist
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = 'images'
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        # Plot c_ss, n_ss, b_ss
        # Plot steady-state consumption and savings distributions
        age_pers = np.arange(1, S + 1)
        fig, ax = plt.subplots()
        plt.plot(age_pers, c_ss, marker='D', label='Consumption')
        plt.plot(age_pers, np.append(0, b_ss[:-1]), marker='D',
                 label='Savings')
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
