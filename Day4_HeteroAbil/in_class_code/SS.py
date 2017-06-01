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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def get_SS(args, graph=False):
    (KL_init, beta, sigma, emat, chi_n_vec, l_tilde, b, upsilon, lambdas, S,
     J, alpha, A, delta) = args
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
        cmat = np.zeros((S,J))
        nmat = np.zeros((S,J))
        bmat = np.zeros((S,J))
        c1 = 1.0
        for j in range(J):
            c1_guess = c1
            c1_args = (r, w, beta, sigma, chi_n_vec, l_tilde, b, upsilon,
                       emat[:,j], S)
            results_c1 = opt.root(hh.get_bSp1, c1_guess, args=(c1_args))
            c1 = results_c1.x
            cmat[:,j] = hh.get_recurs_c(c1, r, beta, sigma, S)
            nmat[:,j] = hh.get_n_s(cmat[:,j], w, sigma, chi_n_vec, l_tilde, b,
                              upsilon, emat[:,j])
            bmat[:,j] = hh.get_recurs_b(cmat[:,j], nmat[:,j], r, w, emat[:,j])
        K_new = aggr.get_K(bmat[:-1,:], lambdas, S)
        L_new = aggr.get_L(nmat, emat, lambdas, S)
        KL_new = np.array([K_new, L_new])
        dist = ((KL_new - KL_init) ** 2).sum()
        KL_init = xi * KL_new + (1 - xi) * KL_init
        print('iter:', ss_iter, ' dist: ', dist)

    c_ss = cmat
    n_ss = nmat
    b_ss = bmat
    K_ss = K_new
    L_ss = L_new
    r_ss = r
    w_ss = w
    Y_params = (alpha, A)
    Y_ss = aggr.get_Y(K_ss, L_ss, Y_params)
    C_ss = aggr.get_C(c_ss, lambdas, S)

    ss_output = {'c_ss': c_ss, 'n_ss': n_ss, 'b_ss': b_ss, 'K_ss': K_ss,
                 'L_ss': L_ss, 'r_ss': r_ss, 'w_ss': w_ss, 'Y_ss': Y_ss,
                 'C_ss': C_ss}

    if graph:
        '''
        ----------------------------------------------------------------
        cur_path    = string, path name of current directory
        output_fldr = string, folder in current path to save files
        output_dir  = string, total path of images folder
        output_path = string, path of file name of figure to be saved
        sgrid       = (S,) vector, ages from 1 to S
        lamcumsum   = (J,) vector, cumulative sum of lambdas vector
        jmidgrid    = (J,) vector, midpoints of ability percentile bins
        smat        = (J, S) matrix, sgrid copied down J rows
        jmat        = (J, S) matrix, jmidgrid copied across S columns
        ----------------------------------------------------------------
        '''
        # Create directory if images directory does not already exist
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = 'images'
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        # Plot 3D steady-state consumption distribution
        sgrid = np.arange(1, S + 1)
        lamcumsum = lambdas.cumsum()
        jmidgrid = 0.5 * lamcumsum + 0.5 * (lamcumsum - lambdas)
        smat, jmat = np.meshgrid(sgrid, jmidgrid)
        cmap_c = cm.get_cmap('summer')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel(r'age-$s$')
        ax.set_ylabel(r'ability-$j$')
        ax.set_zlabel(r'indiv. consumption $c_{j,s}$')
        ax.plot_surface(smat, jmat, c_ss.T, rstride=1,
                        cstride=6, cmap=cmap_c)
        output_path = os.path.join(output_dir, 'c_ss_3D')
        plt.savefig(output_path)
        # plt.show()
        plt.close()

        # Plot 2D steady-state consumption distribution
        minorLocator = MultipleLocator(1)
        fig, ax = plt.subplots()
        linestyles = np.array(["-", "--", "-.", ":"])
        markers = np.array(["x", "v", "o", "d", ">", "|"])
        pct_lb = 0
        for j in range(J):
            this_label = (str(int(np.rint(pct_lb))) + " - " +
                          str(int(np.rint(pct_lb + 100 * lambdas[j]))) +
                          "%")
            pct_lb += 100 * lambdas[j]
            if j <= 3:
                ax.plot(sgrid, c_ss[:, j], label=this_label,
                        linestyle=linestyles[j], color='black')
            elif j > 3:
                ax.plot(sgrid, c_ss[:, j], label=this_label,
                        marker=markers[j - 4], color='black')
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel(r'age-$s$')
        ax.set_ylabel(r'indiv. consumption $c_{j,s}$')
        output_path = os.path.join(output_dir, 'c_ss_2D')
        plt.savefig(output_path)
        # plt.show()
        plt.close()


        # Plot 3D steady-state labor supply distribution
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel(r'age-$s$')
        ax.set_ylabel(r'ability-$j$')
        ax.set_zlabel(r'labor supply $n_{j,s}$')
        ax.plot_surface(smat, jmat, n_ss.T, rstride=1,
                        cstride=6, cmap=cmap_c)
        output_path = os.path.join(output_dir, 'n_ss_3D')
        plt.savefig(output_path)
        # plt.show()
        plt.close()

        # Plot 2D steady-state labor supply distribution
        minorLocator = MultipleLocator(1)
        fig, ax = plt.subplots()
        linestyles = np.array(["-", "--", "-.", ":"])
        markers = np.array(["x", "v", "o", "d", ">", "|"])
        pct_lb = 0
        for j in range(J):
            this_label = (str(int(np.rint(pct_lb))) + " - " +
                          str(int(np.rint(pct_lb + 100 * lambdas[j]))) +
                          "%")
            pct_lb += 100 * lambdas[j]
            if j <= 3:
                ax.plot(sgrid, n_ss[:, j], label=this_label,
                        linestyle=linestyles[j], color='black')
            elif j > 3:
                ax.plot(sgrid, n_ss[:, j], label=this_label,
                        marker=markers[j - 4], color='black')
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel(r'age-$s$')
        ax.set_ylabel(r'labor supply $n_{j,s}$')
        output_path = os.path.join(output_dir, 'n_ss_2D')
        plt.savefig(output_path)
        # plt.show()
        plt.close()


        # Plot 3D steady-state savings/wealth distribution
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel(r'age-$s$')
        ax.set_ylabel(r'ability-$j$')
        ax.set_zlabel(r'indiv. savings $b_{j,s}$')
        ax.plot_surface(smat, jmat, b_ss.T, rstride=1,
                        cstride=6, cmap=cmap_c)
        output_path = os.path.join(output_dir, 'b_ss_3D')
        plt.savefig(output_path)
        # plt.show()
        plt.close()

        # Plot 2D steady-state savings/wealth distribution
        fig, ax = plt.subplots()
        linestyles = np.array(["-", "--", "-.", ":"])
        markers = np.array(["x", "v", "o", "d", ">", "|"])
        pct_lb = 0
        for j in range(J):
            this_label = (str(int(np.rint(pct_lb))) + " - " +
                          str(int(np.rint(pct_lb + 100 * lambdas[j]))) +
                          "%")
            pct_lb += 100 * lambdas[j]
            if j <= 3:
                ax.plot(sgrid, b_ss[:, j], label=this_label,
                        linestyle=linestyles[j], color='black')
            elif j > 3:
                ax.plot(sgrid, b_ss[:, j], label=this_label,
                        marker=markers[j - 4], color='black')
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel(r'age-$s$')
        ax.set_ylabel(r'indiv. savings $b_{j,s}$')
        output_path = os.path.join(output_dir, 'b_ss_2D')
        plt.savefig(output_path)
        # plt.show()
        plt.close()


    return ss_output
