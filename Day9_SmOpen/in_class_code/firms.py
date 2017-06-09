'''
This module contains functions related to the firm

This module defines the following function(s):
    get_r()
    get_w()

'''


def get_r(K, L, params):
    alpha, A, delta = params
    r = alpha * A * ((L / K) ** (1 - alpha)) - delta

    return r


def get_w(params):
    alpha, A, delta, r_star = params
    w = ((1 - alpha) * A * ((alpha * A) / (r_star + delta)) ** (alpha /
                                                                (1 - alpha)))

    return w

def get_K_d(L_d, params):
    alpha, A, delta, r_star = params
    K_d = L_d * ((alpha * A) / (r_star + delta)) ** (1 / (1 - alpha))

    return K_d
