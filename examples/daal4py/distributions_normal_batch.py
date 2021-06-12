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

# daal4py normal distribution example for shared memory systems

import daal4py as d4p
import numpy as np


def main(readcsv=None, method='defaultDense'):

    # Create algorithm
    algorithm = d4p.distributions_normal(engine=d4p.engines_mt19937(seed=777))

    # Create array and fill with normal distribution
    data = np.zeros((1, 10))
    res = algorithm.compute(data)

    assert(np.allclose(data, res.randomNumbers))
    assert(np.allclose(
        data,
        [[
            -0.74104167, -0.13616829, -0.13679562, 2.40385531, -0.33556821,
            0.19041699, -0.61331181, 0.95958821, -0.42301092, 0.09460208
        ]]
    ))

    return data


if __name__ == "__main__":
    res = main()
    print("\nNormal distribution output:", res)
    print("All looks good!")
