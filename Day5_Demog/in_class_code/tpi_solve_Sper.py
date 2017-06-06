# Import packages
import numpy as np
import scipy.optimize as opt
import demographics as demog

# Declare parameters
E = 20
S = 80
T = 4 * S
beta_annual = .96
beta = beta_annual ** (80 / S)
sigma = 3.0
nvec = np.ones(S)
nvec[round(2*S/3):] = 0.2

# Firm parameters
A = 1.0
alpha = .35
delta_annual = .05
delta = 1 - ((1 - delta_annual) ** (80/S))

# Population parameters
min_age = 1
max_age = 100
start_year = 2017
pop_graphs = False
(omega_path_S, g_n_SS, omega_SS, surv_rates_S, rho_s, g_n_path,
    imm_rates_path, omega_S_preTP) = demog.get_pop_objs(E, S, T,
                                                        min_age,
                                                        max_age,
                                                        start_year,
                                                        pop_graphs)
imm_rates_SS = imm_rates_path[-1, :]

# Economic growth
g_y = 0.03

def get_Y(K, L, params):
    (alpha, A) = params
    Y = A * (K ** alpha) * (L ** (1 - alpha))

    return Y


def get_C(cvec, omega_SS):
    C = (omega_SS * cvec).sum()

    return C


def get_I(K, Kp1, bvec, params):
    (delta, g_n_SS, omega_SS, imm_rates_SS, g_y) = params
    capital_flow = ((1 + g_n_SS) * np.exp(g_y) * (imm_rates_SS[1:] *
                    omega_SS[1:] * bvec).sum())

    I = ((1 + g_n_SS) * np.exp(g_y) * Kp1 - (1.0 - delta) * K -
         capital_flow)

    return I


def get_K(bvec, params):
    '''
    bvec = (S-1,) vector of savings

    '''
    (g_n_SS, omega_SS, imm_rates_SS) = params
    K = ((1 / (1 + g_n_SS)) * (omega_SS[:-1] * bvec + imm_rates_SS[1:] *
                               omega_SS[1:] * bvec).sum())

    return K


def get_L(nvec, omega_SS):
    L = (omega_SS * nvec).sum()

    return L


def get_BQ(bvec, r, params):
    (g_n_SS, omega_SS, rho_s) = params
    BQ = (((1 + r) / (1 + g_n_SS)) * (rho_s[:-1] * omega_SS[:-1] *
                                      bvec).sum())

    return BQ


def get_r(K, L, params):
    alpha, A, delta = params
    r = alpha * A * ((L / K) ** (1 - alpha)) - delta

    return r


def get_w(K, L, params):
    alpha, A = params
    w = (1 - alpha) * A * ((K / L) ** alpha)

    return w


def EulerFunc(bvec, *args):
    (nvec, beta, sigma, alpha, A, delta, g_n_SS, omega_SS, imm_rates_SS,
     rho_s, g_y) = args
    b_s = np.append(0.0, bvec)
    b_sp1 = np.append(bvec, 0.0)
    K_params = (g_n_SS, omega_SS, imm_rates_SS)
    K = get_K(bvec, K_params)
    L = get_L(nvec, omega_SS)
    r_params = (alpha, A, delta)
    r = get_r(K, L, r_params)
    w_params = (alpha, A)
    w = get_w(K, L, w_params)
    BQ_params = (g_n_SS, omega_SS, rho_s)
    BQ = get_BQ(bvec, r, BQ_params)
    c = (1 + r) * b_s + w * nvec - np.exp(g_y) * b_sp1 + BQ
    MU_c = c**(-sigma)
    errors = (MU_c[:-1] - (np.exp(-sigma * g_y) * (1 - rho_s[:-1]) *
                           beta * (1 + r) * MU_c[1:]))

    return errors


def EulerFuncT(bvec, *args):
    # p = number of periods left in life time
    # bvec = length p-1
    # rpath = length p
    # nvec = length p
    # wpath = length p
    b_init, nvec, rpath, wpath, BQpath, rho_s, beta, sigma, g_y = args
    b_s = np.append(b_init, bvec)
    b_sp1 = np.append(bvec, 0.0)
    c = (1 + rpath) * b_s + wpath * nvec + BQpath - np.exp(g_y) * b_sp1
    MU_c = c ** (-sigma)
    errors = (MU_c[:-1] - np.exp(-sigma * g_y) * beta * (1 + rpath[1:]) *
              (1 - rho_s[:-1]) * MU_c[1:])

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


# Find SS solution
bvec_init = np.ones(S - 1) * 0.1
b_args = (nvec, beta, sigma, alpha, A, delta, g_n_SS, omega_SS,
          imm_rates_SS, rho_s, g_y)
results_b = opt.root(EulerFunc, bvec_init, args=(b_args))
print(results_b)
b_ss = results_b.x
print('SS savings: ', b_ss)
K_params = (g_n_SS, omega_SS, imm_rates_SS)
K_ss = get_K(b_ss, K_params)
L_ss = get_L(nvec, omega_SS)
print('K_ss and L_ss', np.array([K_ss, L_ss]))
r_params = (alpha, A, delta)
w_params = (alpha, A)
r_ss = get_r(K_ss, L_ss, r_params)
w_ss = get_w(K_ss, L_ss, w_params)
print('SS r and w: ', np.array([r_ss, w_ss]))
b_s_ss = np.append(0.0, b_ss)
b_sp1_ss = np.append(b_ss, 0.0)
BQ_params = (g_n_SS, omega_SS, rho_s)
BQ_ss = get_BQ(b_ss, r_ss, BQ_params)
print('SS BQ: ', BQ_ss)
c_ss = (1 + r_ss) * b_s_ss + w_ss * nvec - np.exp(g_y) * b_sp1_ss + BQ_ss
print('SS consumption: ', c_ss)
Y_params = (alpha, A)
Y_ss = get_Y(K_ss, L_ss, Y_params)
C_ss = get_C(c_ss, omega_SS)
I_params = (delta, g_n_SS, omega_SS, imm_rates_SS, g_y)
I_ss = get_I(K_ss, K_ss, b_ss, I_params)
RC_error = Y_ss - C_ss - I_ss
print('RC Error: ', RC_error)



# Start TPI solution
b1vec = 1.05 * b_ss
K_params = (g_n_path[0], omega_S_preTP, imm_rates_path[0, :])
K1 = get_K(b1vec, K_params)
Kpath_init = np.zeros(T + S - 1)
Kpath_init[:T] = np.linspace(K1, K_ss, T)
Kpath_init[T:] = K_ss

r_params = (A, alpha, delta)
r1 = get_r(K1, L_ss, r_params)

BQ_params = (g_n_path[0], omega_S_preTP, rho_s)
BQ1 = get_BQ(b1vec, r1, BQ_params)
BQpath_init = np.zeros(T + S - 1)
BQpath_init[:T] = np.linspace(BQ1, BQ_ss, T)
BQpath_init[T:] = BQ_ss

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
    for p in range(2, S):
        bvec_guess = np.diagonal(bmat[S - p:, :p - 1])
        b_init = bmat[S - p - 1, 0]
        b_args = (b_init, nvec[-p:], rpath[:p], wpath[:p],
                  BQpath_init[:p], rho_s[-p:], beta, sigma, g_y)
        results_bp = opt.root(EulerFuncT, bvec_guess, args=(b_args))
        b_sol_p = results_bp.x
        DiagMaskbp = np.eye(p - 1, dtype=bool)
        bmat[S - p:, 1:p] = DiagMaskbp * b_sol_p + bmat[S - p:, 1:p]

    for t in range(1, T + 1):  # Go from periods 1 to S+T-1
        bvec_guess = np.diagonal(bmat[:, t - 1:t + S - 2])
        b_init = 0.0
        b_args = (b_init, nvec, rpath[t - 1:t + S - 1],
                  wpath[t - 1:t + S - 1], BQpath_init[t - 1:t + S - 1],
                  rho_s, beta, sigma, g_y)
        results_bt = opt.root(EulerFuncT, bvec_guess, args=(b_args))
        b_sol_t = results_bt.x
        DiagMaskbt = np.eye(S - 1, dtype=bool)
        bmat[:, t:t + S - 1] = (DiagMaskbt * b_sol_t +
                                bmat[:, t:t + S - 1])

    Kpath_new = np.zeros(T)
    BQpath_new = np.zeros(T)
    Kpath_new[0] = K1
    BQpath_new[0] = BQ1
    Kpath_new[1:] = \
        (1 / (1 + g_n_path[1:T])) * (omega_path_S[:T - 1, :-1] *
         bmat[:, 1:T].T + imm_rates_path[:T - 1, 1:] *
         omega_path_S[:T - 1, 1:] * bmat[:, 1:T].T).sum(axis=1)
    BQpath_new[1:] = \
        ((1 + rpath[1:T]) / (1 + g_n_path[1:T])) * (rho_s[:-1] *
         omega_path_S[:T - 1, :-1] *
         bmat[:, 1:T].T).sum(axis=1)
    KBQ_init = np.append(Kpath_init[:T], BQpath_init[:T])
    KBQ_new = np.append(Kpath_new[:T], BQpath_new[:T])
    dist = ((KBQ_init - KBQ_new) ** 2).sum()
    Kpath_init[:T] = xi * Kpath_new[:T] + (1 - xi) * Kpath_init[:T]
    BQpath_init[:T] = xi * BQpath_new[:T] + (1 - xi) * BQpath_init[:T]
    print('iter:', tpi_iter, ' dist: ', dist)
