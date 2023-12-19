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

# daal4py covariance example for streaming on shared memory systems

from pathlib import Path

from readcsv import pd_read_csv, read_next

import daal4py as d4p


def main(readcsv=pd_read_csv, *args, **kwargs):
    data_path = Path(__file__).parent / "data" / "batch"
    infile = data_path / "covcormoments_dense.csv"

    # configure a covariance object
    algo = d4p.covariance(streaming=True)

    # get the generator (defined in stream.py)...
    rn = read_next(infile, 112, readcsv)
    # ... and iterate through chunks/stream
    for chunk in rn:
        algo.compute(chunk)

    # finalize computation
    result = algo.finalize()

    return result


if __name__ == "__main__":
    res = main()
    print("Covariance matrix:\n", res.covariance)
    print("Mean vector:\n", res.mean)
    print("All looks good!")
