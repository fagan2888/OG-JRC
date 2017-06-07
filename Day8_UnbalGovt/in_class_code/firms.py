'''
This module contains functions related to the firm

This module defines the following function(s):
    get_r()
    get_w()

'''


def get_r(K, L, params):
    alpha, A, delta, tau_c = params
    r = (1-tau_c) * (alpha * A * ((L / K) ** (1 - alpha)) - delta)

    return r


def get_w(K, L, params):
    alpha, A = params
    w = (1 - alpha) * A * ((K / L) ** alpha)

    return w
