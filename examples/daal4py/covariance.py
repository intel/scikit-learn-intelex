# ==============================================================================
# Copyright 2014 Intel Corporation
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

# daal4py covariance example for shared memory systems

from pathlib import Path

import numpy as np
from readcsv import pd_read_csv

import daal4py as d4p


def main(readcsv=pd_read_csv, method="defaultDense"):
    data_path = Path(__file__).parent / "data" / "batch"
    infile = data_path / "covcormoments_dense.csv"

    # configure a covariance object
    algo = d4p.covariance()

    # let's provide a file directly, not a table/array
    result1 = algo.compute(str(infile))

    # We can also load the data ourselfs and provide the numpy array
    algo = d4p.covariance(method=method)
    data = readcsv(infile)
    _ = algo.compute(data)

    # covariance result objects provide correlation, covariance and mean
    assert np.allclose(result1.covariance, result1.covariance)
    assert np.allclose(result1.mean, result1.mean)
    assert np.allclose(result1.correlation, result1.correlation)

    return result1


if __name__ == "__main__":
    res = main()
    print("Covariance matrix:\n", res.covariance)
    print("Mean vector:\n", res.mean)
    print("All looks good!")
