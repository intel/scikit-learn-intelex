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

# daal4py cholesky example for shared memory systems

from pathlib import Path

import numpy as np

import daal4py as d4p
from daal4py.sklearn.utils import pd_read_csv


def main(readcsv=pd_read_csv):
    data_path = Path(__file__).parent / "data" / "batch"
    infile = data_path / "cholesky.csv"

    # configure a cholesky object
    algo = d4p.cholesky()

    # let's provide a file directly, not a table/array
    return algo.compute(str(infile))
    # cholesky result objects provide choleskyFactor


if __name__ == "__main__":
    result = main()
    print("\nFactor:\n", result.choleskyFactor)
    print("All looks good!")
