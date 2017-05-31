# Import packages
import numpy as np
import scipy.optimize as opt

# Declare parameters
S = 3
beta_annual = .96
beta = beta_annual ** 20
sigma = 3.0
nvec = np.array([1.0, 1.0, 0.2])

# Firm parameters
A = 1.0
alpha = .35
delta_annual = .05
delta = 1 - ((1 - delta_annual) ** 20)


def get_K(bvec):
    K = bvec.sum()

    return K


def get_L(nvec):
    L = nvec.sum()

    return L


def get_r(K, L, params):
    alpha, A, delta = params
    r = alpha * A * ((L / K) ** (1 - alpha)) - delta

    return r


def get_w(K, L, params):
    alpha, A = params
    w = (1 - alpha) * A * ((K / L) ** alpha)

    return w


def EulerFunc(bvec, *args):
    b2, b3 = bvec
    nvec, beta, sigma, alpha, A, delta = args
    K = get_K(bvec)
    L = get_L(nvec)
    r_params = (alpha, A, delta)
    r = get_r(K, L, r_params)
    w_params = (alpha, A)
    w = get_w(K, L, w_params)
    c1 = w * nvec[0] - b2
    c2 = (1 + r) * b2 + w * nvec[1] - b3
    c3 = (1 + r) * b3 + w * nvec[2]
    MU_c1 = c1 ** (-sigma)
    MU_c2 = c2 ** (-sigma)
    MU_c3 = c3 ** (-sigma)
    error1 = MU_c1 - beta * (1 + r) * MU_c2
    error2 = MU_c2 - beta * (1 + r) * MU_c3
    errors = np.array([error1, error2])

    return errors


def EulerFuncT(bvec, *args):
    b2, b3 = bvec
    nvec, rpath, wpath, beta, sigma = args
    r2, r3 = rpath
    w1, w2, w3 = wpath
    c1 = w1 * nvec[0] - b2
    c2 = (1 + r2) * b2 + w2 * nvec[1] - b3
    c3 = (1 + r3) * b3 + w3 * nvec[2]
    MU_c1 = c1 ** (-sigma)
    MU_c2 = c2 ** (-sigma)
    MU_c3 = c3 ** (-sigma)
    error1 = MU_c1 - beta * (1 + r2) * MU_c2
    error2 = MU_c2 - beta * (1 + r3) * MU_c3
    errors = np.array([error1, error2])

    return errors


def LoneEuler(b3, *args):
    b2, nvec, rpath, wpath, beta, sigma = args
    n2, n3 = nvec
    r1, r2 = rpath
    w1, w2 = wpath
    c2 = (1 + r1) * b2 + w1 * n2 - b3
    c3 = (1 + r2) * b3 + w2 * n3
    MU_c2 = c2 ** (-sigma)
    MU_c3 = c3 ** (-sigma)
    error = MU_c2 - beta * (1 + r2) * MU_c3

    return error


bvec_init = np.array([0.1, 0.1])
b_args = (nvec, beta, sigma, alpha, A, delta)
results_b = opt.root(EulerFunc, bvec_init, args=(b_args))
print(results_b)
b_ss = results_b.x
b2_ss, b3_ss = b_ss
print('SS savings: ', b_ss)
K_ss = get_K(b_ss)
L_ss = get_L(nvec)
print('K_ss and L_ss', np.array([K_ss, L_ss]))
r_params = (alpha, A, delta)
w_params = (alpha, A)
r_ss = get_r(K_ss, L_ss, r_params)
w_ss = get_w(K_ss, L_ss, w_params)
print('SS r and w: ', np.array([r_ss, w_ss]))
c1_ss = w_ss * nvec[0] - b2_ss
c2_ss = (1 + r_ss) * b2_ss + w_ss * nvec[1] - b3_ss
c3_ss = (1 + r_ss) * b3_ss + w_ss * nvec[2]
print('SS c1, c2, c3: ', np.array([c1_ss, c2_ss, c3_ss]))
    

# Start TPI solution
T = 50
b1vec = 1.05 * b_ss
K1 = get_K(b1vec)
Kpath_init = np.zeros(T + 1)
Kpath_init[:-1] = np.linspace(K1, K_ss, T)
Kpath_init[-1] = K_ss

dist = 10
mindist = 1e-08
maxiter = 500
tpi_iter = 0
xi = 0.2

r_params = (alpha, A, delta)
w_params = (alpha, A)

while dist > mindist and tpi_iter < maxiter:
    tpi_iter += 1
    # Get r and w paths
    rpath = get_r(Kpath_init, L_ss, r_params)
    wpath = get_w(Kpath_init, L_ss, w_params)
    bmat = np.zeros((S - 1, T + 1))
    bmat[:, 0] = b1vec
    # Solve for the lone individual problem in period 1
    b32_init = b1vec[1]
    b32_args = (b1vec[0], nvec[1:], rpath[:2], wpath[:2], beta, sigma)
    results_b32 = opt.root(LoneEuler, b32_init, args=(b32_args))
    bmat[1, 1] = results_b32.x
    for t in range(T - 1):
        bvec_init = np.array([bmat[0, t], bmat[1, t + 1]])
        b_args = (nvec, rpath[t + 1:t + 3], wpath[t:t + 3], beta, sigma)
        results_bt = opt.root(EulerFuncT, bvec_init, args=(b_args))
        b2, b3 = results_bt.x
        bmat[0, t + 1] = b2
        bmat[1, t + 2] = b3

    Kpath_new = bmat.sum(axis=0)
    dist = ((Kpath_init[:-1] - Kpath_new[:-1]) ** 2).sum()
    Kpath_init[:-1] = xi * Kpath_new[:-1] + (1 - xi) * Kpath_init[:-1]
    print('iter:', tpi_iter, ' dist: ', dist)
