'''
This module defines functions for the tax rates and total
tax Liability.
'''

# import libararies


# define tax functions
def get_taxrates(lab_inc, cap_inc, factor, params):
    (A, B, C, D, max_x, min_x, max_y, min_y, shift_x, shift_y, shift,
     phi) = params

    X = lab_inc * factor
    Y = cap_inc * factor

    X2 = X ** 2
    Y2 = Y ** 2

    ratio_x = (A * X2 + B * X) / (A * X2 + B * X + 1)
    tau_x = (max_x - min_x) * ratio_x + min_x

    ratio_y = (A * Y2 + B * Y) / (A * Y2 + B * Y + 1)
    tau_y = (max_y - min_y) * ratio_y + min_y

    tau = ((tau_x + shift_x) ** phi) * ((tau_y + shift_y) ** (1-phi)) + shift

    return tau


def get_tot_tax_liab(lab_inc, cap_inc, factor, etr_params):
    etr = get_taxrates(lab_inc, cap_inc, factor, etr_params)
    tot_tax_liab = etr * (lab_inc + cap_inc)

    return tot_tax_liab
