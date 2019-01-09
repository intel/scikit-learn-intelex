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

# daal4py PCA example for shared memory systems

import daal4py as d4p
import numpy as np

# let's try to use pandas' fast csv reader
try:
    import pandas
    read_csv = lambda f, c=None, t=np.float64: pandas.read_csv(f, usecols=c, delimiter=',', header=None, dtype=t)
except:
    # fall back to numpy loadtxt
    read_csv = lambda f, c=None, t=np.float64: np.loadtxt(f, usecols=c, delimiter=',', ndmin=2)


def main(readcsv=read_csv, method='svdDense'):
    infile = "./data/batch/pca_normalized.csv"

    # configure a PCA object
    algo = d4p.pca(resultsToCompute="mean|variance|eigenvalue", isDeterministic=True)
    
    # let's provide a file directly, not a table/array
    result1 = algo.compute(infile)

    # We can also load the data ourselfs and provide the numpy array
    data = readcsv(infile)
    result2 = algo.compute(data)

    # PCA result objects provide eigenvalues, eigenvectors, means and variances
    assert np.allclose(result1.eigenvalues, result2.eigenvalues)
    assert np.allclose(result1.eigenvectors, result2.eigenvectors)
    assert np.allclose(result1.means, result2.means)
    assert np.allclose(result1.variances, result2.variances)
    assert result1.eigenvalues.shape == (1, data.shape[1])
    assert result1.eigenvectors.shape == (data.shape[1], data.shape[1])
    assert result1.means.shape == (1, data.shape[1])
    assert result1.variances.shape == (1, data.shape[1])

    return result1


if __name__ == "__main__":
    result1 = main()
    print("\nEigenvalues:\n", result1.eigenvalues)
    print("\nEigenvectors:\n", result1.eigenvectors)
    print("\nMeans:\n", result1.means)
    print("\nVariances:\n", result1.variances)
    print('All looks good!')
