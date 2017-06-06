'''
This script will estimate a tax function
'''

# import
import pandas as pd
import scipy.optimize as opt
import numpy as np
import pickle

import tax_func_graphs as graph_tax

# statistical objective function
# takes in data
# takes in intial guesses at parameters
# returns a value of the obj function
def crit(params_init, *args):
    # (A_tilde, B_tilde, C_tilde, D_tilde, max_x, max_y, phi,
    #  shift) = params_init
    # min_x, min_y, shift_x, shift_y, df = args
    # estimate different parameters
    (A_tilde, B_tilde, C_tilde, D_tilde, max_x, max_y, phi) = params_init
    min_x, min_y, shift, df = args
    ## Change here - needed weighted average for X and Y....
    # X = ((df['Labor Income'] - df['Labor Income'].mean()) /
    #      df['Labor Income'].mean())
    # Y = ((df['Capital Income'] - df['Capital Income'].mean()) /
    #      df['Capital Income'].mean())
    # X2 = ((df['Labor Income'] ** 2 - (df['Labor Income'] ** 2).mean()) /
    #      (df['Labor Income'] ** 2).mean())
    # Y2 = ((df['Capital Income'] ** 2 - (df['Capital Income'] ** 2).mean()) /
    #      (df['Capital Income'] ** 2).mean())
    X = df['Labor Income']
    Y = df['Capital Income']
    wgts = df['Weights']
    tax_rates = df['MTRx']

    shift_x = np.maximum(-min_x, 0.0) + 0.01 * (max_x - min_x)
    shift_y = np.maximum(-min_y, 0.0) + 0.01 * (max_y - min_y)

    # X_tilde = ((X - ((wgts * X).sum() / wgts.sum())) / ((wgts * X).sum()
    #                                                     / wgts.sum()))
    # X2_tilde = (((X ** 2) - ((wgts * (X ** 2)).sum() / wgts.sum())) /
    #             ((wgts * (X ** 2)).sum() / wgts.sum()))
    # Y_tilde = ((Y - ((wgts * Y).sum() / wgts.sum())) / ((wgts * Y).sum()
    #                                                     / wgts.sum()))
    # Y2_tilde = (((Y ** 2) - ((wgts * (Y ** 2)).sum() / wgts.sum())) /
    #             ((wgts * (Y ** 2)).sum() / wgts.sum()))

    X2 = X ** 2
    Y2 = Y ** 2
    X2bar = (X2 * wgts).sum() / wgts.sum()
    Xbar = (X * wgts).sum() / wgts.sum()
    Y2bar = (Y2 * wgts).sum() / wgts.sum()
    Ybar = (Y * wgts).sum() / wgts.sum()
    shift_x = np.maximum(-min_x, 0.0) + 0.01 * (max_x - min_x)
    shift_y = np.maximum(-min_y, 0.0) + 0.01 * (max_y - min_y)
    X2_tilde = (X2 - X2bar) / X2bar
    X_tilde = (X - Xbar) / Xbar
    Y2_tilde = (Y2 - Y2bar) / Y2bar
    Y_tilde = (Y - Ybar) / Ybar

    E_tilde = A_tilde + B_tilde
    F_tilde = C_tilde + D_tilde

    ratio_x = ((A_tilde * X2_tilde + B_tilde * X_tilde + E_tilde) /
               (A_tilde * X2_tilde + B_tilde * X_tilde + E_tilde + 1))
    # this was -min_x...
    tau_x = (max_x - min_x) * ratio_x + min_x

    ratio_y = ((C_tilde * Y2_tilde + D_tilde * Y_tilde + F_tilde) /
               (C_tilde * Y2_tilde + D_tilde * Y_tilde + F_tilde + 1))
    # this was -min_y
    tau_y = (max_y - min_y) * ratio_y + min_y

    tau_est = (((tau_x + shift_x) ** phi) * ((tau_y + shift_y) ** (1-phi)) +
               shift)

    #print('Params guess: ', params_init)
    crit = (wgts * ((tau_est - tax_rates) ** 2)).sum()

    return crit



# Read in data
micro_data = pickle.load(open('../taxdata42.pkl','rb'))
df2017 = micro_data['2017']

# compute some variables
df2017['MTRx'] = (((df2017['MTR wage'] * df2017['Wage and Salaries']) +
                (df2017['MTR self-employed Wage'] *
                 np.absolute(df2017['Self-Employed Income'])))
                  / (df2017['Wage and Salaries'] +
                     np.absolute(df2017['Self-Employed Income'])))

df2017['ETR'] = df2017['Total Tax Liability'] / df2017['Adjusted Total income']

df2017.rename(columns={"MTR capital income": "MTRy",
                       "Wage + Self-Employed Income": "Labor Income",
                       "Adjusted Total income": "ATI"}, inplace=True)


df2017['Capital Income'] = df2017['ATI'] - df2017['Labor Income']

# clean data
# drop obs with < $5 income
df2017.drop(df2017[df2017['ATI'] < 5.0].index, inplace=True)
df2017.drop(df2017[df2017['Capital Income'] < 5.0].index, inplace=True)
df2017.drop(df2017[df2017['Labor Income'] < 5.0].index, inplace=True)

# drop obs with very high ETR
# df2017.drop(df2017[df2017['ETR'] > 70.0].index, inplace=True)
df2017.drop(df2017[df2017['ETR'] > 65.0].index, inplace=True)

# drop obs with very low ETR
# df2017.drop(df2017[df2017['ETR'] < -30.0].index, inplace=True)
df2017.drop(df2017[df2017['ETR'] < -15.0].index, inplace=True)

# drop obs with very low MTR
df2017.drop(df2017[df2017['MTRx'] > 99.0].index, inplace=True)
df2017.drop(df2017[df2017['MTRy'] > 99.0].index, inplace=True)

# drop obs with very high MTR
# df2017.drop(df2017[df2017['MTRx'] < -50.0].index, inplace=True)
# df2017.drop(df2017[df2017['MTRy'] < -50.0].index, inplace=True)
df2017.drop(df2017[df2017['MTRx'] < -45.0].index, inplace=True)
df2017.drop(df2017[df2017['MTRy'] < -45.0].index, inplace=True)





# call an a minimizer
# define params fixed outside estimation
y10 = df2017['Capital Income'].quantile(0.1)
x10 = df2017['Labor Income'].quantile(0.1)
min_x = (df2017['MTRx'][df2017['Capital Income'] < y10]).min()
min_y = (df2017['MTRx'][df2017['Labor Income'] < x10]).min()
shift_x = np.absolute(min_x) - 0.01
shift_y = np.absolute(min_y) - 0.01


# make initial guesses for estimated parameters
# A_tilde_init = 0.1
# B_tilde_init = 0.1
# C_tilde_init = 0.1
# D_tilde_init = 0.1
# max_x_init = 0.7
# max_y_init = 0.7
# phi_init = 0.6
shift_init = 0.0

y20 = df2017['Capital Income'].quantile(0.2)
x20 = df2017['Labor Income'].quantile(0.2)

A_tilde_init = 1.0
B_tilde_init = 1.0
C_tilde_init = 1.0
D_tilde_init = 1.0
phi_init = 0.5
max_x_init = np.minimum(
    df2017['MTRx'][(df2017['Capital Income'] < y20)].max(), 0.7)
max_y_init = np.minimum(
    df2017['MTRx'][(df2017['Labor Income'] < x20)].max(), 0.7)


shift = df2017['MTRx'][(df2017['Labor Income'] < x20) |
    (df2017['Capital Income'] < y20)].min()

# params_init = np.array([A_tilde_init, B_tilde_init, C_tilde_init,
#                         D_tilde_init, max_x_init, max_y_init,
#                         phi_init, shift_init])
# est_args = (min_x, min_y, shift_x, shift_y, df2017)
# bnds = ((1e-12, None), (1e-12, None),
#         (1e-12, None), (1e-12, None), (min_x + 1e-4, 1.0), (min_y + 1e-4, 1.0),
#         (0, 1), (None, None))
params_init = np.array([A_tilde_init, B_tilde_init, C_tilde_init,
                        D_tilde_init, max_x_init, max_y_init,
                        phi_init])
est_args = (min_x, min_y, shift, df2017)
bnds = ((1e-12, None), (1e-12, None),
        (1e-12, None), (1e-12, None), (min_x + 1e-4, 1.0), (min_y + 1e-4, 1.0),
        (0, 1))
lb_max_x = np.maximum(min_x, 0.0) + 1e-4
lb_max_y = np.maximum(min_y, 0.0) + 1e-4
# bnds = ((1e-12, None), (1e-12, None), (1e-12, None), (1e-12, None),
#     (lb_max_x, 0.8), (lb_max_y, 0.8), (0, 1))
params_hat = opt.minimize(crit, params_init, args=(est_args), method='L-BFGS-B',
                          bounds=bnds, tol=1e-15)
print('Parameter estimates: ', params_hat)

# transform back into parameters A, B, C...
X = df2017['Labor Income']
Y = df2017['Capital Income']
wgts = df2017['Weights']
# X_bar = (wgts * X).sum() / wgts.sum()
# X2_bar = (wgts * (X ** 2)).sum() / wgts.sum()
# Y_bar = (wgts * Y).sum() / wgts.sum()
# Y2_bar = (wgts * (Y ** 2)).sum() / wgts.sum()
X2 = X ** 2
Y2 = Y ** 2
X2_bar = (X2 * wgts).sum() / wgts.sum()
X_bar = (X * wgts).sum() / wgts.sum()
Y2_bar = (Y2 * wgts).sum() / wgts.sum()
Y_bar = (Y * wgts).sum() / wgts.sum()


final_params = np.zeros(12)
max_x = params_hat.x[4]
max_y = params_hat.x[5]
phi = params_hat.x[6]
# shift = params_hat.x[7]
shift_x = np.maximum(-min_x, 0.0) + 0.01 * (max_x - min_x)
shift_y = np.maximum(-min_y, 0.0) + 0.01 * (max_y - min_y)
final_params[:4] = ((params_hat.x[:4]) / np.array([X2_bar, X_bar, Y2_bar,
                                                   Y_bar]))
final_params[4:] = np.array([max_x, min_x, max_y, min_y, shift_x,
    shift_y, shift, phi])
print('Parameter estimates: ', final_params)

graphs = True
if graphs:
    final3D = np.reshape(final_params,(1,1,12))
    graph_tax.plot_mtr(final3D, 0, 0, 'labor')
    quit()
