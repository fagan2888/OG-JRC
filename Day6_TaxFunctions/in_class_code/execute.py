'''
This script executes the steady-state solution
'''

import numpy as np
import SS as ss
import elliputil as elp
import txfunc_est as txf
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
# chose ellipse parameters to match constant Frisch elasticity function
b_ellip_init = 1.0
upsilon_init = 0.0
ellip_init = np.array([b_ellip_init, upsilon_init])
Frisch_elast = 0.8
CFE_scale = 1.0
cfe_params = np.array([Frisch_elast, CFE_scale])
b_ellip, upsilon = elp.fit_ellip_CFE(ellip_init, cfe_params, l_tilde, False)
# b = 0.501
# upsilon = 1.553

## Tax function parameters
num_tx_params = 12
#micro_data = pickle.load(open('../taxdata42.pkl','rb'))
baseline = True
analytical_mtrs = False
age_specific = True
start_year = 2017
txf_dict = txf.tax_func_estimate(start_year, baseline, analytical_mtrs,
                                 age_specific)
tax_dict = pickle.dump(txf_dict, open('tax_dict.pkl', 'wb'))
# txf_dict = pickle.load(open('tax_dict.pkl', 'rb'))
etrparam_vec = txf_dict['tfunc_etr_params_S'].reshape(num_tx_params)
mtrxparam_vec = txf_dict['tfunc_mtrx_params_S'].reshape(num_tx_params)
mtryparam_vec = txf_dict['tfunc_mtry_params_S'].reshape(num_tx_params)
avg_income_data = txf_dict['tfunc_avginc']

etrparam_vec[6] = 0.0
mtrxparam_vec[6] = 0.0
mtryparam_vec[6] = 0.0



# Firm parameters
A = 1.0
alpha = 0.35
delta_annual = 0.05
delta = 1 - ((1 - delta_annual) ** (80 / S))
# SS parameters
SS_graph = True


K_init = 100.0
L_init = 50.0
X_init = 0.4 * A * (K_init ** alpha) * (L_init ** (1 - alpha)) / S
factor_init = (avg_income_data / (A * (K_init ** alpha) * (L_init **
                                                           (1 - alpha)) / S))
init_vals = np.array([K_init, L_init, X_init, factor_init])

ss_args = (init_vals, beta, sigma, chi_n_vec, l_tilde, b_ellip, upsilon, S,
           alpha, A, delta, etrparam_vec, mtrxparam_vec, mtryparam_vec,
           avg_income_data)
ss_output = ss.get_SS(ss_args, SS_graph)
pickle.dump(ss_output, open('ss_output.pkl', 'wb'))
