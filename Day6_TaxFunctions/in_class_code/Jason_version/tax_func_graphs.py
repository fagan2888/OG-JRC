

'''
This module allows for plotting tax functions in 2D

'''

# import packages
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_mtr(mtr_params, age, year, income_type):

    other_income = np.array([0.0,1000., 10000., 100000., 1000000.]) # fix capital income
    N = 1000 # number of points in income grids
    inc_sup = np.linspace(5, 200000, N)

    A = mtr_params[age,year,0]
    B = mtr_params[age,year,1]
    C = mtr_params[age,year,2]
    D = mtr_params[age,year,3]
    max_x = mtr_params[age,year,4]
    min_x = mtr_params[age,year,5]
    max_y = mtr_params[age,year,6]
    min_y = mtr_params[age,year,7]
    shift_x = mtr_params[age,year,8]
    shift_y = mtr_params[age,year,9]
    shift = mtr_params[age,year,10]
    share = mtr_params[age,year,11]

    marginal_rates = np.zeros((other_income.shape[0],N))

    for i in range((other_income.shape[0])):
        if income_type == 'labor':
            Y = other_income[i] # fix other income
            X = inc_sup # income of income_type varies
            label_string = 'Capital'
        else:
            Y = other_income[i] # fix other income
            X = inc_sup # income of income_type varies
            label_string = 'Labor'

        X2 = X ** 2
        Y2 = Y ** 2
        tau_x = ((max_x - min_x) * (A * X2 + B * X) /
            (A * X2 + B * X + 1) + min_x)
        tau_y = ((max_y - min_y) * (C * Y2 + D * Y) /
            (C * Y2 + D * Y + 1) + min_y)
        tau = (((tau_x + shift_x) ** share) *
            ((tau_y + shift_y) ** (1 - share))) + shift

        marginal_rates[i,:]  =  tau

        plt.plot(inc_sup, marginal_rates[0,:], label=label_string+' income = '+str(other_income[i]))

    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    plt.legend(loc='center right')
    plt.title('MTR by '+income_type+' income')
    plt.xlabel(r'Income')
    plt.ylabel(r'MTR')
    output_path = os.path.join(output_dir, 'mtr_'+income_type)
    plt.savefig(output_path)
    # plt.show()
    plt.close()
