#===============================================================================
# Copyright 2014-2021 Intel Corporation
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
#===============================================================================

# daal4py covariance example for distributed memory systems; SPMD mode
# run like this:
#    mpirun -n 4 python ./covariance_spmd.py

import daal4py as d4p

# let's use a reading of file in chunks (defined in spmd_utils.py)
from spmd_utils import read_csv, get_chunk_params


def main():
    infile = "./data/batch/covcormoments_dense.csv"

    # We know the number of lines in the file
    # and use this to separate data between processes
    skiprows, nrows = get_chunk_params(lines_count=200,
                                       chunks_count=d4p.num_procs(),
                                       chunk_number=d4p.my_procid())

    # Each process reads its chunk of the file
    data = read_csv(infile, sr=skiprows, nr=nrows)

    # Create algorithm with distributed mode
    alg = d4p.covariance(method="defaultDense", distributed=True)

    # Perform computation
    res = alg.compute(data)

    # covariance result objects provide correlation, covariance and mean
    assert res.covariance.shape == (data.shape[1], data.shape[1])
    assert res.mean.shape == (1, data.shape[1])
    assert res.correlation.shape == (data.shape[1], data.shape[1])

    return res


if __name__ == "__main__":
    # Initialize SPMD mode
    d4p.daalinit()
    res = main()
    # result is available on all processes - but we print only on root
    if d4p.my_procid() == 0:
        print(res)
    d4p.daalfini()
