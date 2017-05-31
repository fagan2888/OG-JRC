# Import packages
import numpy as np
import scipy.optimize as opt

# Declare parameters
S = 80
beta_annual = .96
beta = beta_annual ** (80/S)
sigma = 3.0
nvec = np.ones(S)
nvec[round(2*S/3):] = 0.2

# Firm parameters
A = 1.0
alpha = .35
delta_annual = .05
delta = 1 - ((1 - delta_annual) ** (80/S))


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
    nvec, beta, sigma, alpha, A, delta = args
    b_s = np.append(0.0,bvec)
    b_sp1 = np.append(bvec,0.0)
    K = get_K(bvec)
    L = get_L(nvec)
    r_params = (alpha, A, delta)
    r = get_r(K, L, r_params)
    w_params = (alpha, A)
    w = get_w(K, L, w_params)
    c = (1+r)*b_s + w*nvec - b_sp1
    MU_c = c**(-sigma)
    errors = MU_c[:-1] - beta*(1+r)*MU_c[1:] 

    return errors


def EulerFuncT(bvec, *args):
    # p = number of periods left in life time
    # bvec = length p-1
    # rpath = length p
    # nvec = length p
    # wpath = length p
    b_init, nvec, rpath, wpath, beta, sigma = args
    b_s = np.append(b_init, bvec)
    b_sp1 = np.append(bvec,0.0)
    c = (1+rpath)*b_s + wpath*nvec - b_sp1
    MU_c = c**(-sigma)
    errors = MU_c[:-1] - beta*(1+rpath[1:])*MU_c[1:]

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


bvec_init = np.ones(S-1)*0.1
b_args = (nvec, beta, sigma, alpha, A, delta)
results_b = opt.root(EulerFunc, bvec_init, args=(b_args))
print(results_b)
b_ss = results_b.x
print('SS savings: ', b_ss)
K_ss = get_K(b_ss)
L_ss = get_L(nvec)
print('K_ss and L_ss', np.array([K_ss, L_ss]))
r_params = (alpha, A, delta)
w_params = (alpha, A)
r_ss = get_r(K_ss, L_ss, r_params)
w_ss = get_w(K_ss, L_ss, w_params)
print('SS r and w: ', np.array([r_ss, w_ss]))
b_s_ss = np.append(0.0,b_ss)
b_sp1_ss = np.append(b_ss, 0.0)
c_ss = (1+r_ss)*b_s_ss + w_ss * nvec - b_sp1_ss
print('SS consumption: ', c_ss)
    

# Start TPI solution
T = 3*S
b1vec = 1.05 * b_ss
K1 = get_K(b1vec)
Kpath_init = np.zeros(T + S - 1)
Kpath_init[:T] = np.linspace(K1, K_ss, T)
Kpath_init[T:] = K_ss

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
    bmat = np.zeros((S - 1, T + S - 1))
    bmat[:, 0] = b1vec
    # Solve for households' problems
    for p in range(2,S): 
        bvec_guess = np.diagonal(bmat[S - p:, :p - 1]) 
        b_init = bmat[S-p-1,0]
        b_args = (b_init, nvec[-p:], rpath[:p], wpath[:p], beta, sigma)
        results_bp = opt.root(EulerFuncT, bvec_guess, args=(b_args))
        b_sol_p = results_bp.x
        DiagMaskbp = np.eye(p - 1, dtype=bool)
        bmat[S - p:, 1:p] = DiagMaskbp * b_sol_p + bmat[S - p:, 1:p]
 

        
    for t in range(1, T+1):  # Go from periods 1 to S+T-1
        bvec_guess = np.diagonal(bmat[:, t - 1:t + S - 2])
        b_init = 0.0
        b_args = (b_init, nvec, rpath[t - 1:t + S - 1], wpath[t - 1:t + S - 1],
                  beta, sigma)
        results_bt = opt.root(EulerFuncT, bvec_guess, args=(b_args))
        b_sol_t = results_bt.x
        DiagMaskbt = np.eye(S-1, dtype=bool)
        bmat[:, t:t+S-1] = DiagMaskbt * b_sol_t + bmat[:, t:t+S-1]

    
    Kpath_new = bmat.sum(axis=0)
    dist = ((Kpath_init[:T] - Kpath_new[:T]) ** 2).sum()
    Kpath_init[:T] = xi * Kpath_new[:T] + (1 - xi) * Kpath_init[:T]
    print('iter:', tpi_iter, ' dist: ', dist)
