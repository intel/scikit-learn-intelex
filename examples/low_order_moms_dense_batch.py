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
    read_csv = lambda f, c: pandas.read_csv(f, usecols=c, delimiter=',').values
except:
    # fall back to numpy loadtxt
    read_csv = lambda f, c: np.loadtxt(f, usecols=c, delimiter=',')


def main():
    # read data from file
    file = "./data/batch/covcormoments_dense.csv"
    data = read_csv(file, range(10))

    # compute
    alg = d4p.low_order_moments()
    res = alg.compute(data)

    # result provides minimum, maximum, sum, sum of squares,
    # sum of squared difference from the means, mean,
    # second order raw moment, variance, standard deviation, variation
    assert res.minimum.shape == (1, 10) \
        and res.maximum.shape == (1, 10) \
        and res.sum.shape == (1, 10) \
        and res.sumSquares.shape == (1, 10) \
        and res.sumSquaresCentered.shape == (1, 10) \
        and res.mean.shape == (1, 10) \
        and res.secondOrderRawMoment.shape == (1, 10) \
        and res.variance.shape == (1, 10) \
        and res.standardDeviation.shape == (1, 10) \
        and res.variation.shape == (1, 10)


if __name__ == "__main__":
    main()
    print('All looks good!')
