'''
This script executes the steady-state solution
'''

import numpy as np
import SS as ss
import elliputil as elp
import ability as abil
import pickle

# Declare parameters

# Household parameters
S = int(80)
beta_annual = 0.96
beta = beta_annual ** (80 / S)
sigma = 2.5
l_tilde = 1.0
chi_n_vec = 1.0 * np.ones(S)
# chi_n_vec = np.hstack((np.linspace(1.2, 1.0, 10),
#                        np.linspace(1.0, 1.0, 40),
#                        np.linspace(1.0, 3.0, 30)))
# b = 0.501
# upsilon = 1.553
ellip_init = np.array([0.2, 1.0])
Frisch = 0.8
scale_param = 1.0
cfe_params = np.array([Frisch, scale_param])
b, upsilon = elp.fit_ellip_CFE(ellip_init, cfe_params, l_tilde, True)

lambdas = np.array([0.3, 0.3, 0.2, 0.1, 0.1])
J = lambdas.shape[0]
age_wgts = np.ones(S)*(1/S)
age_wgts_80 = np.ones(80)*(1/80)
emat = abil.get_e_interp(S, age_wgts, age_wgts_80, lambdas, True)

# Firm parameters
A = 1.0
alpha = 0.35
delta_annual = 0.05
delta = 1 - ((1 - delta_annual) ** (80 / S))
# SS parameters
SS_graph = True


K_init = 100.0
L_init = 50.0
KL_init = np.array([K_init, L_init])

ss_args = (KL_init, beta, sigma, emat, chi_n_vec, l_tilde, b, upsilon, lambdas,
           S, J, alpha, A, delta)
ss_output = ss.get_SS(ss_args, SS_graph)
pickle.dump(ss_output, open('ss_output.pkl', 'wb'))
