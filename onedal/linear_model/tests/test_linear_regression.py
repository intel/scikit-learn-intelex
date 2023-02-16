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
    from numpy.testing import assert_array_equal, assert_allclose

    from onedal.linear_model import LinearRegression
    from onedal.tests.utils._device_selection import get_queues

    from sklearn.datasets import load_diabetes
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    @pytest.mark.parametrize('queue', get_queues())
    def test_diabetes(queue):
        X, y = load_diabetes(return_X_y=True)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y,
                             train_size=0.8, random_state=777)
        model = LinearRegression(fit_intercept=True)
        model.fit(X_train, y_train, queue=queue)
        y_pred = model.predict(X_test, queue=queue)
        assert mean_squared_error(y_test, y_pred) < 2396

    @pytest.mark.parametrize('queue', get_queues())
    def test_pickle(queue):
        assert len(get_queues())
        X, y = load_diabetes(return_X_y=True)
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
    def test_full_results(queue):
        seed = 42
        f_count, r_count = 19, 7
        s_count, t_count = 3500, 1999

        np.random.seed(seed)
        intp = np.random.rand(r_count)
        coef = np.random.rand(r_count, f_count).T

        X = np.random.rand(s_count, f_count)
        y = X @ coef + intp[np.newaxis, :]

        model = LinearRegression(fit_intercept=True)
        model.fit(X, y, queue=queue)

        assert_allclose(coef, model.coef_.T)
        assert_allclose(intp, model.intercept_)

        Xt = np.random.rand(t_count, f_count)
        gtr = Xt @ coef + intp[np.newaxis, :]

        res = model.predict(Xt, queue=queue)

        assert_allclose(gtr, res)

    @pytest.mark.parametrize('queue', get_queues())
    def test_no_intercept_results(queue):
        seed = 42
        f_count, r_count = 19, 7
        s_count, t_count = 3500, 1999

        np.random.seed(seed)
        coef = np.random.rand(r_count, f_count).T

        X = np.random.rand(s_count, f_count)
        y = X @ coef

        model = LinearRegression(fit_intercept=True)
        model.fit(X, y, queue=queue)

        assert_allclose(coef, model.coef_.T)

        Xt = np.random.rand(t_count, f_count)
        gtr = Xt @ coef

        res = model.predict(Xt, queue=queue)

        assert_allclose(gtr, res)

    @pytest.mark.parametrize('queue', get_queues())
    def test_reconstruct_model(queue):
        seed = 42
        s_count = 3500
        f_count, r_count = 14, 9

        np.random.seed(seed)
        intp = np.random.rand(r_count)
        coef = np.random.rand(r_count, f_count).T

        X = np.random.rand(s_count, f_count)
        gtr = X @ coef + intp[np.newaxis, :]

        model = LinearRegression(fit_intercept=True)
        model.coef_ = coef.T
        model.intercept_ = intp

        res = model.predict(X, queue=queue)
        from onedal.datatypes._data_conversion import from_table
        print(from_table(model._onedal_model.packed_coefficients))
        assert_allclose(gtr, res)
