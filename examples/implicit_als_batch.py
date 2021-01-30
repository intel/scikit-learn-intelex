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

# daal4py implicit_als example for shared memory systems

import daal4py as d4p
import numpy as np

# let's try to use pandas' fast csv reader
try:
    import pandas

    def read_csv(f, c=None, t=np.float64):
        return pandas.read_csv(f, usecols=c, delimiter=',', header=None, dtype=t)
except ImportError:
    # fall back to numpy loadtxt
    def read_csv(f, c=None, t=np.float64):
        return np.loadtxt(f, usecols=c, delimiter=',', ndmin=2)


def main(readcsv=read_csv, method='defaultDense'):
    nFactors = 2
    infile = "./data/batch/implicit_als_dense.csv"
    # We load the data
    data = readcsv(infile)

    # configure a implicit_als init object
    algo1 = d4p.implicit_als_training_init(nFactors=nFactors, method=method)
    # and compute initial model
    result1 = algo1.compute(data)

    # configure a implicit_als training object
    algo2 = d4p.implicit_als_training(nFactors=nFactors, method=method)
    # and compute model using initial model
    result2 = algo2.compute(data, result1.model)

    # Now do some prediction; first get prediction algorithm object
    algo3 = d4p.implicit_als_prediction_ratings(nFactors=nFactors)
    # and compute
    result3 = algo3.compute(result2.model)

    # implicit als prediction result objects provide prediction
    assert(result3.prediction.shape == data.shape)

    return result3


if __name__ == "__main__":
    res = main()
    print("Predicted ratings:\n", res.prediction[:10])
    print('All looks good!')
