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
    from numpy.testing import assert_allclose, assert_array_equal

    from onedal.linear_model import LinearRegression
    from onedal.tests.utils._device_selection import get_queues

    from sklearn.datasets import load_diabetes
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    @pytest.mark.parametrize('queue', get_queues())
    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    def test_diabetes(queue, dtype):
        X, y = load_diabetes(return_X_y=True)
        X, y = X.astype(dtype), y.astype(dtype)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y,
                             train_size=0.8, random_state=777)
        model = LinearRegression(fit_intercept=True)
        model.fit(X_train, y_train, queue=queue)
        y_pred = model.predict(X_test, queue=queue)
        assert mean_squared_error(y_test, y_pred) < 2396

    @pytest.mark.parametrize('queue', get_queues())
    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    def test_pickle(queue, dtype):
        X, y = load_diabetes(return_X_y=True)
        X, y = X.astype(dtype), y.astype(dtype)
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y, queue=queue)
        expected = model.predict(X, queue=queue)

        import pickle
        dump = pickle.dumps(model)
        model2 = pickle.loads(dump)

        assert isinstance(model2, model.__class__)
        result = model2.predict(X, queue=queue)

        assert_array_equal(expected, result)

    @pytest.mark.parametrize('queue', get_queues())
    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    def test_full_results(queue, dtype):
        seed = 42
        f_count, r_count = 19, 7
        s_count, t_count = 3500, 1999

        gen = np.random.default_rng(seed)
        intp = gen.random(size=r_count, dtype=dtype)
        coef = gen.random(size=(r_count, f_count), dtype=dtype).T

        X = gen.random(size=(s_count, f_count), dtype=dtype)
        y = X @ coef + intp[np.newaxis, :]

        model = LinearRegression(fit_intercept=True)
        model.fit(X, y, queue=queue)

        tol = 2e-3 if model.coef_.dtype == np.float32 else 1e-5
        assert_allclose(coef, model.coef_.T, rtol=tol)

        tol = 2e-3 if model.intercept_.dtype == np.float32 else 1e-5
        assert_allclose(intp, model.intercept_, rtol=tol)

        Xt = gen.random(size=(t_count, f_count), dtype=dtype)
        gtr = Xt @ coef + intp[np.newaxis, :]

        res = model.predict(Xt, queue=queue)

        tol = 2e-4 if res.dtype == np.float32 else 1e-7
        assert_allclose(gtr, res, rtol=tol)

    @pytest.mark.parametrize('queue', get_queues())
    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    def test_no_intercept_results(queue, dtype):
        seed = 42
        f_count, r_count = 19, 7
        s_count, t_count = 3500, 1999

        gen = np.random.default_rng(seed)
        coef = gen.random(size=(r_count, f_count), dtype=dtype).T

        X = gen.random(size=(s_count, f_count), dtype=dtype)
        y = X @ coef

        model = LinearRegression(fit_intercept=False)
        model.fit(X, y, queue=queue)

        tol = 2e-3 if model.coef_.dtype == np.float32 else 1e-7
        assert_allclose(coef, model.coef_.T, rtol=tol)

        Xt = gen.random(size=(t_count, f_count), dtype=dtype)
        gtr = Xt @ coef

        res = model.predict(Xt, queue=queue)

        tol = 5e-5 if res.dtype == np.float32 else 1e-7
        assert_allclose(gtr, res, rtol=tol)

    @pytest.mark.parametrize('queue', get_queues())
    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    def test_reconstruct_model(queue, dtype):
        seed = 42
        s_count = 3500
        f_count, r_count = 14, 9

        gen = np.random.default_rng(seed)
        intp = gen.random(size=r_count, dtype=dtype)
        coef = gen.random(size=(r_count, f_count), dtype=dtype).T

        X = gen.random(size=(s_count, f_count), dtype=dtype)
        gtr = X @ coef + intp[np.newaxis, :]

        model = LinearRegression(fit_intercept=True)
        model.coef_ = coef.T
        model.intercept_ = intp

        res = model.predict(X, queue=queue)

        tol = 1e-5 if res.dtype == np.float32 else 1e-7
        assert_allclose(gtr, res, rtol=tol)
