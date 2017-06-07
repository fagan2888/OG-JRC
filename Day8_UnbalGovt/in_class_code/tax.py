
# import packages
import aggregates as aggr

def get_revenue(K, L, r, w, nvec, bvec, params):
    alpha, A, delta, tau_k, tau_l, tau_c = params
    Y_params = (alpha, A)
    Y = aggr.get_Y(K, L , Y_params)
    cit_rev = tau_c + (Y - w * L) - tau_c * delta * K
    iit_rev = (tau_l * w * nvec).sum() + (tau_k * r * bvec[:-1]).sum()
    R = cit_rev + iit_rev
    return R
