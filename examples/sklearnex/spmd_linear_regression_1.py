import numpy as np

from mpi4py import MPI
from dpctl import SyclQueue
from sklearnex.spmd.linear_model import LinearRegression

def generate_X_y(par, data_seed):
    ns, nf, nr = par['ns'], par['nf'], par['nr']

    crng = np.random.default_rng(777)
    coef = crng.uniform(-4, 1, size=(nr, nf)).T
    intp = crng.uniform(-1, 9, size=(nr, ))

    drng = np.random.default_rng(data_seed)
    data = drng.uniform(-7, 7, size=(ns, nf))
    resp = data @ coef + intp[np.newaxis, :]

    return data, resp

def get_train_data(rank):
    par = {'ns': 128, 'nf': 8, 'nr': 4}
    return generate_X_y(par, rank)

def get_test_data(rank):
    par = {'ns': 1024, 'nf': 8, 'nr': 4}
    return generate_X_y(par, rank)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

X, y = get_train_data(rank)

queue = SyclQueue("gpu")

model = LinearRegression().fit(X, y, queue)

print(f"Coefficients on rank {rank}:\n", model.coef_)
print(f"Intercept on rank {rank}:\n", model.intercept_)

X_test, _ = get_test_data(rank)

result = model.predict(X_test, queue)

print(f"Result on rank {rank}:\n", result)
