import pytest
import numpy as np
import pandas as pd
from sklearn.utils._testing import assert_array_equal
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from daal4py.sklearn import patch_sklearn, unpatch_sklearn


def make_dataset(n_samples, n_features, kind=np.array, random_state=0):
    try:
        from pandas import DataFrame
        if kind not in (list, np.array, DataFrame):
            kind = np.array
    except ImportError:
        if kind not in (list, np.array):
            kind = np.array

    x, y = make_regression(n_samples, n_features, random_state=random_state)

    if kind == list:
        x = list(x)
        for i, row in enumerate(x):
            x[i] = list(row)
        y = list(y)

    try:
        from pandas import DataFrame

        if kind == DataFrame:
            x = DataFrame(data=x, index=None, columns=None)
            y = DataFrame(y)
    except ImportError:
        pass

    return x, y


def test_linear_array_vs_dataframe_homogen():
    pd = pytest.importorskip('pandas')

    x_train, y_train = make_dataset(100, 20)
    x_test, _ = make_dataset(100, 20, random_state=1)

    df_x_train, df_y_train = make_dataset(100, 20, pd.DataFrame)
    df_x_test, _ = make_dataset(100, 20, pd.DataFrame, random_state=1)

    array_reg = LinearRegression()
    array_reg.fit(x_train, y_train)

    df_reg = LinearRegression()
    df_reg.fit(x_train, y_train)

    assert_array_equal(array_reg.coef_, df_reg.coef_)
    assert_array_equal(array_reg.intercept_, df_reg.intercept_)
    assert_array_equal(array_reg.predict(x_test), df_reg.predict(x_test))


def test_linear_array_vs_dataframe_heterogen():
    pd = pytest.importorskip('pandas')

    x_train, y_train = make_dataset(100, 20)
    x_test, _ = make_dataset(100, 20, random_state=1)

    df_x_train, df_y_train = make_dataset(100, 20, pd.DataFrame)
    df_x_test, _ = make_dataset(100, 20, pd.DataFrame, random_state=1)

    dir_dtypes = {col: 'float64' if i % 2 == 0 else 'float32' for i, col in enumerate(df_x_train.columns)}
    df_x_train = df_x_train.astype(dir_dtypes)
    df_x_test = df_x_test.astype(dir_dtypes)

    array_reg = LinearRegression()
    array_reg.fit(x_train, y_train)

    df_reg = LinearRegression()
    df_reg.fit(x_train, y_train)

    assert_array_equal(array_reg.coef_, df_reg.coef_)
    assert_array_equal(array_reg.intercept_, df_reg.intercept_)
    assert_array_equal(array_reg.predict(x_test), df_reg.predict(x_test))
