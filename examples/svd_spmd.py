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

# daal4py SVD example for distributed memory systems; SPMD mode
# run like this:
#    mpirun -n 4 python ./svd_spmd.py

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


def main():
    infile = "./data/batch/svd.csv"
    # We know the number of lines in the file and use this to separate data between processes
    lines_count = 16000
    block_size = (int)(lines_count/d4p.num_procs()) + 1
    # Last process reads the file to the end
    data = read_csv(infile, sr=d4p.my_procid()*block_size, nr=block_size)

    # configure a SVD object
    algo = d4p.svd(distributed=True)

    # We can also load the data ourselfs and provide the numpy array
    result = algo.compute(data)

    # SVD result objects provide leftSingularMatrix, rightSingularMatrix and singularValues
    assert result.singularValues.shape == (1, data.shape[1])
    assert result.rightSingularMatrix.shape == (data.shape[1], data.shape[1])

    # leftSingularMatrix not yet supported in dist mode
    # TODO: remove condition after adding this support
    if result.leftSingularMatrix is not None:
        assert result.leftSingularMatrix.shape == data.shape
        self.assertTrue(np.allclose(data, np.matmul(np.matmul(result.leftSingularMatrix,np.diag(result.singularValues[0])),result.rightSingularMatrix)))

    return data, result


if __name__ == "__main__":
    # Initialize SPMD mode
    d4p.daalinit()
    data, result = main()
    print("results in process number", d4p.my_procid())
    print(result)
    print('All looks good!')
    d4p.daalfini()
