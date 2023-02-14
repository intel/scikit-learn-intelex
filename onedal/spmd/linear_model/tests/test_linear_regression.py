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

from daal4py.sklearn._utils import daal_check_version, sklearn_check_version

if daal_check_version((2023, 'P', 100)):
    import pytest
    import numpy as np

    from mpi4py import MPI

    from numpy.testing import assert_allclose

    from onedal.spmd.linear_model import LinearRegression
    from onedal.tests.utils._device_selection import get_queues

    from sklearn.datasets import load_diabetes
    from sklearn.metrics import mean_squared_error

    def split_datasets(n_ranks, *arrays):
        first = arrays[0]
        n_samples = first.shape[1]

        percentage = 1 / float(n_ranks)
        block = int(n_samples * percentage)
        assert int(block * n_ranks) <= n_samples

        n_arrays = len(arrays)
        results = ([],) * n_arrays

        for b in range(n_ranks):
            for a in range(n_arrays):
                first = block * b
                last = min(first + block, n_samples)
                shard = arrays[a][:, first : last]
                results[a].append(shard)

        return results

    def run_spmd_training(queue, params, X, y):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.get_size()

        Xs, ys = split_datasets(X, y)

        model = LinearRegression(**params)
        model.fit(Xs[rank], ys[rank], queue)

        coef = np.array(model.coef_)

        test = (size + rank) % size 
        Xt, yt = Xs[test], yt[test]

        yp = model.predict(Xt, queue)
        yp = np.array(yp)

        return mean_squared_error(yt, yp)
                
    @pytest.mark.parametrize('queue', get_queues())
    def test_diabetes(queue):
        X, y = load_diabetes(return_X_y=True)

        model = LinearRegression(fit_intercept=True)
        model.fit(X_train, y_train, queue=queue)
        y_pred = model.predict(X_test, queue=queue)
        assert mean_squared_error(y_test, y_pred) < 2396



