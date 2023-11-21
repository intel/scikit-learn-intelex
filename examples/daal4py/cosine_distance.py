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

# daal4py cosine distance example for shared memory systems

import os
from pathlib import Path

import numpy as np

import daal4py as d4p
from daal4py.sklearn.utils import pd_read_csv


def main(readcsv=pd_read_csv):
    data_path = Path(__file__).parent.parent / "daal4py" / "data" / "batch"
    data_file = data_path / "distance.csv"
    data = readcsv(data_file, usecols=range(10))

    # Create algorithm to compute cosine distance (no parameters)
    algorithm = d4p.cosine_distance()

    # Computed cosine distance with file or numpy array
    res1 = algorithm.compute(str(data_file))
    res2 = algorithm.compute(data)

    assert np.allclose(res1.cosineDistance, res2.cosineDistance)

    return res1


if __name__ == "__main__":
    res = main()
    print("\nCosine distance (first 15 rows/columns):\n", res.cosineDistance[0:15, 0:15])
    print("All looks good!")
