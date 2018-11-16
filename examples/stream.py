#*******************************************************************************
# Copyright 2014-2018 Intel Corporation
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

# Example showing daal4py's operation on streams using a generator

import daal4py as d4p
import numpy as np
import os

# a generator which reads a file in chunks
def read_next(file, chunksize):
    assert os.path.isfile(file)
    s = 0
    while True:
        # if found a smaller chunk we set s to < 0 to indicate eof
        if s < 0:
            return
        a = np.genfromtxt(file, delimiter=',', skip_header=s, max_rows=chunksize)
        if a.shape[0] == 0:
            return
        if a.ndim == 1:
            a = a[:, np.newaxis]
        # last chunk is usually smaller, if not, numpy will print warning in next iteration
        if chunksize > a.shape[0]:
            s = -1
        else:
            s += a.shape[0]
        yield a

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
