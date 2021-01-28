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

# daal4py PCA example for distributed memory systems; SPMD mode
# run like this:
#    mpirun -n 4 python ./pca_spmd.py

import daal4py as d4p
from numpy import loadtxt, allclose

if __name__ == "__main__":
    # Initialize SPMD mode
    d4p.daalinit()

    # Each process gets its own data
    infile = "./data/distributed/pca_normalized_" + str(d4p.my_procid() + 1) + ".csv"

    # configure a PCA object to use svd instead of default correlation
    algo = d4p.pca(method='svdDense', distributed=True)
    # let's provide a file directly, not a table/array
    result1 = algo.compute(infile)

    # We can also load the data ourselfs and provide the numpy array
    data = loadtxt(infile, delimiter=',')
    result2 = algo.compute(data)

    # PCA result objects provide eigenvalues, eigenvectors, means and variances
    assert allclose(result1.eigenvalues, result2.eigenvalues)
    assert allclose(result1.eigenvectors, result2.eigenvectors)
    assert result1.means is None and \
           result2.means is None or \
           allclose(result1.means, result2.means)
    assert result1.variances is None and \
           result2.variances is None or \
           allclose(result1.variances, result2.variances)

    print('All looks good!')
    d4p.daalfini()
