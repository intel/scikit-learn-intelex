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

# daal4py DBSCAN example for shared memory systems

import daal4py as d4p
import numpy as np

# let's try to use pandas' fast csv reader
try:
    import pandas

    def read_csv(f, c, t=np.float64):
        return pandas.read_csv(f, usecols=c, delimiter=',', header=None, dtype=t)
except ImportError:
    # fall back to numpy loadtxt
    def read_csv(f, c, t=np.float64):
        return np.loadtxt(f, usecols=c, delimiter=',', ndmin=2)


def main(readcsv=read_csv, method='defaultDense'):
    infile = "./data/batch/dbscan_dense.csv"
    epsilon = 0.04
    minObservations = 45

    # Load the data
    data = readcsv(infile, range(2))

    # configure dbscan main object:
    # we also request the indices and observations of cluster cores
    algo = d4p.dbscan(
        minObservations=minObservations,
        epsilon=epsilon,
        resultsToCompute='computeCoreIndices|computeCoreObservations'
    )
    # and compute
    result = algo.compute(data)

    # Note: we could have done this in just one line:
    # assignments = d4p.dbscan(
    #     minObservations=minObservations,
    #     epsilon=epsilon,
    #     resultsToCompute='computeCoreIndices|computeCoreObservations'
    # ).compute(data).assignments

    # DBSCAN result objects provide assignments,
    # nClusters and coreIndices/coreObservations (if requested)
    assert result.assignments.shape == (data.shape[0], 1)
    assert result.coreObservations.shape == (result.coreIndices.shape[0], data.shape[1])

    return result


if __name__ == "__main__":
    result = main()
    print("\nFirst 10 cluster assignments:\n", result.assignments[0:10])
    print("\nFirst 10 cluster core indices:\n", result.coreIndices[0:10])
    print("\nFirst 10 cluster core observations:\n", result.coreObservations[0:10])
    print("\nNumber of clusters:\n", result.nClusters)
    print('All looks good!')
