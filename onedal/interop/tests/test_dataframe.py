# ==============================================================================
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
# ==============================================================================

import numpy as np
import pandas as pd

import pytest

from .wrappers import get_dtype_list

from onedal.interop.array import from_array
import onedal.interop.dataframes as dataframe

dimensions = [
    (1, 1),
    (1, 11),
    (17, 1),
    (3, 7),
    (21, 47),
    (1001, 1),
    (999, 1),
    (888, 777),
    (123, 999),
    (999, 999),
    (1e6, 127)
]

def generate_dtypes(gen, col_count):
    all_dtypes = np.asarray(get_dtype_list())
    indices = gen.integers(len(all_dtypes), size=col_count)
    return np.take(all_dtypes, indices)

def generate_one(gen, dtype, count):
    raw = gen.integers(count, size=count)
    return raw.astype(dtype=dtype)

def generate_dict(gen, row_count, col_count):
    dtypes = generate_dtypes(gen, col_count)
    generate = lambda dt: generate_one(gen, dt, row_count)
    return {str(i): generate(dtypes[i]) for i in range(col_count)}

def generate_dataset(row_count, col_count, seed = None):
    seed = row_count * col_count if seed is None else seed
    generator = np.random.Generator(np.random.MT19937(seed))
    raw_dict = generate_dict(generator, row_count, col_count)
    return pd.DataFrame(raw_dict)

@pytest.mark.parametrize("shape", dimensions)
def test_pandas_conversion(shape):
    row_count, col_count = shape
    row_count = int(row_count)
    col_count = int(col_count)

    df = generate_dataset(row_count, col_count)
    onedal_table = dataframe.to_table(df)

    assert onedal_table.get_column_count() == col_count
    assert onedal_table.get_row_count() == row_count

    for col in range(col_count):
        gtr_column = df.iloc[:, col]
        raw_column = onedal_table.get_column(col)
        column = from_array(raw_column.flatten())
        assert column.dtype == gtr_column.dtype
        np.testing.assert_equal(column, gtr_column)

