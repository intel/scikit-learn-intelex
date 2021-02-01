# ===============================================================================
# Copyright 2021 Intel Corporation
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

import unittest
import tracemalloc
from daal4py.sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np


class Test(unittest.TestCase):
    def gen_clsf_data(self):
        data, label = make_classification(
            n_samples=2000, n_features=50, random_state=777)
        return data, label, \
            data.size * data.dtype.itemsize + label.size * label.dtype.itemsize

    def kfold_function_template(self, data_transform_function):
        tracemalloc.start()

        x, y, data_memory_size = self.gen_clsf_data()
        kf = KFold(n_splits=10)
        x, y = data_transform_function(x, y)

        mem_before, _ = tracemalloc.get_traced_memory()
        for train_index, test_index in kf.split(x):
            if isinstance(x, np.ndarray):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]
            elif isinstance(x, pd.core.frame.DataFrame):
                x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            knn = KNeighborsClassifier()
            knn.fit(x_train, y_train)
        del knn, x_train, x_test, y_train, y_test
        mem_after, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.assertTrue(
            mem_after - mem_before < 0.25 * data_memory_size,
            'Size of extra allocated memory is greater than 25% of input data')

    def test_memory_leak_ndarray_c(self):
        self.kfold_function_template(lambda x, y: (x, y))

    def test_memory_leak_ndarray_f(self):
        self.kfold_function_template(lambda x, y: (np.asfortranarray(x), y))

    def test_memory_leak_dataframe_c(self):
        self.kfold_function_template(
            lambda x, y: (pd.DataFrame(x), pd.Series(y)))

    def test_memory_leak_dataframe_f(self):
        self.kfold_function_template(lambda x, y: (
            pd.DataFrame(np.asfortranarray(x)), pd.Series(y)))


if __name__ == '__main__':
    unittest.main()
