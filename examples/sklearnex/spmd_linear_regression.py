import numpy as np

from mpi4py import MPI

import dpctl

from numpy.testing import assert_allclose

from onedal.spmd.linear_model import LinearRegression as LinRegSpmd

from sklearn.datasets import make_regression

def generate_X_y(par, coef_seed, data_seed):
    ns, nf, nr = par['ns'], par['nf'], par['nr']

    crng = np.random.default_rng(coef_seed)
    coef = crng.uniform(-4, 1, size = (nr, nf)).T
    intp = crng.uniform(-1, 9, size = (nr, ))

    drng = np.random.default_rng(data_seed)
    data = drng.uniform(-7, 7, size = (ns, nf))
    resp = data @ coef + intp[np.newaxis, :]

    return data, resp, coef, intp

if __name__ == "__main__":
    q = dpctl.SyclQueue("gpu")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    params_spmd = {'ns': 15, 'nf': 21, 'nr': 23}
    params_grtr = {'ns': 30, 'nf': 21, 'nr': 23}
    
    Xsp, ysp, csp, isp = generate_X_y(params_spmd, size, rank)
    Xgt, ygt, cgt, igt = generate_X_y(params_grtr, size, rank)

    assert_allclose(csp, cgt)
    assert_allclose(isp, igt)

    lrsp = LinRegSpmd(copy_X=True, fit_intercept=True)
    lrsp.fit(Xsp, ysp, queue=q)

    assert_allclose(lrsp.coef_, csp.T)
    assert_allclose(lrsp.intercept_, isp)

    ypr = lrsp.predict(Xgt, queue=q)

    assert_allclose(ypr, ygt)

    print("Groundtruth responses:\n", ygt)
    print("Computed responses:\n", ypr)
