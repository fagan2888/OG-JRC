'''
This module contains functions related to the firm

This module defines the following function(s):
    get_K()
    get_L()
    get_Y()
    get_C()

'''

import numpy as np


def get_K(bmat, lambdas, S):
    lambda_mat = np.tile(lambdas,(S-1,1))
    K = (lambda_mat*bmat).sum()

    return K


def get_L(nmat, emat, lambdas, S):
    lambda_mat = np.tile(lambdas,(S,1))
    L = (emat*lambda_mat*nmat).sum()

    return L


def get_Y(K, L, params):
    alpha, A = params
    Y = A * (K ** alpha) * (L ** (1 - alpha))

    return Y


def get_C(cmat, lambdas, S):
    lambda_mat = np.tile(lambdas,(S,1))
    C = (lambda_mat*cmat).sum()

    return C
