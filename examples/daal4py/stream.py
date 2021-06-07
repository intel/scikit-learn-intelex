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

# Example showing daal4py's operation on streams using a generator

import daal4py as d4p
import numpy as np
import os

try:
    import pandas

    def read_csv(f, c=None, s=0, n=None, t=np.float64):
        return pandas.read_csv(f, usecols=c, delimiter=',', header=None,
                               skiprows=s, nrows=n, dtype=t)
except:
    # fall back to numpy genfromtxt
    def read_csv(f, c=None, s=0, n=np.iinfo(np.int64).max):
        a = np.genfromtxt(f, usecols=c, delimiter=',', skip_header=s, max_rows=n)
        if a.shape[0] == 0:
            raise Exception("done")
        if a.ndim == 1:
            return a[:, np.newaxis]
        return a


# a generator which reads a file in chunks
def read_next(file, chunksize, readcsv=read_csv):
    assert os.path.isfile(file)
    s = 0
    while True:
        # if found a smaller chunk we set s to < 0 to indicate eof
        if s < 0:
            return
        a = read_csv(file, s=s, n=chunksize)
        # last chunk is usually smaller, if not,
        # numpy will print warning in next iteration
        if chunksize > a.shape[0]:
            s = -1
        else:
            s += a.shape[0]
        yield a


if __name__ == "__main__":
    # get the generator
    rn = read_next("./data/batch/svd.csv", 112)

    # creat an SVD algo object
    algo = d4p.svd(streaming=True)

    # iterate through chunks/stream
    for chunk in rn:
        algo.compute(chunk)

    # finalize computation
    res = algo.finalize()
    print("Singular values:\n", res.singularValues)
