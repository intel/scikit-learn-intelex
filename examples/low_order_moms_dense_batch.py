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

# daal4py low order moments example for shared memory systems

import daal4py as d4p
import numpy as np

# let's try to use pandas' fast csv reader
try:
    import pandas
    read_csv = lambda f, c: pandas.read_csv(f, usecols=c, delimiter=',', header=None, dtype=np.float64)
except:
    # fall back to numpy loadtxt
    read_csv = lambda f, c: np.loadtxt(f, usecols=c, delimiter=',', ndmin=2)


def main():
    # read data from file
    file = "./data/batch/covcormoments_dense.csv"
    data = read_csv(file, range(10))

    # compute
    alg = d4p.low_order_moments()
    res = alg.compute(data)

    # result provides minimum, maximum, sum, sumSquares, sumSquaresCentered,
    # mean, secondOrderRawMoment, variance, standardDeviation, variation
    assert res.minimum.shape == (1, data.shape[1])
    assert res.maximum.shape == (1, data.shape[1])
    assert res.sum.shape == (1, data.shape[1])
    assert res.sumSquares.shape == (1, data.shape[1])
    assert res.sumSquaresCentered.shape == (1, data.shape[1])
    assert res.mean.shape == (1, data.shape[1])
    assert res.secondOrderRawMoment.shape == (1, data.shape[1])
    assert res.variance.shape == (1, data.shape[1])
    assert res.standardDeviation.shape == (1, data.shape[1])
    assert res.variation.shape == (1, data.shape[1])

    return res


if __name__ == "__main__":
    res = main()
    # print results
    print("\nMinimum:\n", res.minimum)
    print("\nMaximum:\n", res.maximum)
    print("\nSum:\n", res.sum)
    print("\nSum of squares:\n", res.sumSquares)
    print("\nSum of squared difference from the means:\n", res.sumSquaresCentered)
    print("\nMean:\n", res.mean)
    print("\nSecond order raw moment:\n", res.secondOrderRawMoment)
    print("\nVariance:\n", res.variance)
    print("\nStandard deviation:\n", res.standardDeviation)
    print("\nVariation:\n", res.variation)
    print('All looks good!')
