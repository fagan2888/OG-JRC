'''
This module contains functions related to the firm

This module defines the following function(s):
    get_K()
    get_L()
    get_Y()
    get_C()

'''

import numpy as np


def get_K(bvec):
    K = bvec.sum()

    return K


def get_L(nvec):
    L = nvec.sum()

    return L


def get_Y(K, L, params):
    alpha, A = params
    Y = A * (K ** alpha) * (L ** (1 - alpha))

    return Y


def get_C(cvec):
    C = cvec.sum()

    return C
