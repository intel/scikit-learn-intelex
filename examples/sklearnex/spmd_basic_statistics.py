# ===============================================================================
# Copyright 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

import numpy as np
from mpi4py import MPI

import dpctl
from numpy.testing import assert_allclose
from sklearnex.spmd.basic_statistics import BasicStatistics as BasicStatisticsSpmd


def generate_data(par, size, seed=777):
    ns, nf = par['ns'], par['nf']

    data_blocks, weight_blocks = [], []
    rng = np.random.default_rng(seed)

    for b in range(size):
        data = rng.uniform(b, (b + 1) * (b + 1),
                           size=(ns, nf))
        weights = rng.uniform(1, (b + 1), size=ns)
        weight_blocks.append(weights)
        data_blocks.append(data)

    data = np.concatenate(data_blocks, axis=0)
    weights = np.concatenate(weight_blocks)

    return (data, weights)


if __name__ == "__main__":
    q = dpctl.SyclQueue("gpu")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    params_spmd = {'ns': 19, 'nf': 31}

    data, weights = generate_data(params_spmd, size)

    weighted_data = np.diag(weights) @ data

    gtr_mean = np.mean(weighted_data, axis=0)
    gtr_std = np.std(weighted_data, axis=0)

    bss = BasicStatisticsSpmd(["mean", "standard_deviation"])
    res = bss.compute(data, weights, queue=q)

    dtype = res["mean"].dtype
    std_tol = 1e-2 if dtype == np.float32 else 1e-7
    mean_tol = 1e-5 if dtype == np.float32 else 1e-7

    assert_allclose(res["mean"], gtr_mean, rtol=mean_tol)
    assert_allclose(res["standard_deviation"], gtr_std, rtol=std_tol)

    print("Groundtruth mean:\n", gtr_mean)
    print("Computed mean:\n", res["mean"])

    print("Groundtruth std:\n", gtr_std)
    print("Computed std:\n", res["standard_deviation"])
