#*******************************************************************************
# Copyright 2014-2019 Intel Corporation
# All Rights Reserved.
#
# This software is licensed under the Apache License, Version 2.0 (the
# "License"), the following terms apply:
#
# You may not use this file except in compliance with the License.  You may
# obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#*******************************************************************************

# daal4py covariance example for shared memory systems; SPMD mode
# run like this:
#    mpirun -n 4 python ./covariance_spmd.py

import daal4py as d4p
import numpy as np

# let's try to use pandas' fast csv reader
try:
    import pandas
    read_csv = lambda f, c=None, sr=0, nr=None, t=np.float64: pandas.read_csv(f,
                                                                              usecols=c,
                                                                              skiprows=sr,
                                                                              nrows=nr,
                                                                              delimiter=',',
                                                                              header=None,
                                                                              dtype=t)
except:
    # fall back to numpy loadtxt
    def read_csv(f, c=None, sr=0, nr=np.iinfo(np.int64).max, t=np.float64):
        print("sr",sr,"nr",nr)
        res = np.genfromtxt(f,
                      usecols=c,
                      delimiter=',',
                      skip_header=sr,
                      max_rows=nr,
                      dtype=t)
        if res.ndim == 1:
            return res[:, np.newaxis]
        return res


def main(readcsv=read_csv, method='defaultDense'):
    # Each process gets its own data
    infile = "./data/batch/covcormoments_dense.csv"
    # We know the number of lines in the file and use this to separate data between processes
    lines_count = 200
    process_count = 4
    block_size = (int)(lines_count/process_count) + 1
    # Last process reads the file to the end
    data = read_csv(infile, sr=d4p.my_procid()*block_size, nr=block_size)

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
    print("results in process number", d4p.my_procid())
    print(res)
    d4p.daalfini()
