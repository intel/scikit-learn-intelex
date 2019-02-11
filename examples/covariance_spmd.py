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

# daal4py covariance example for shared memory systems; SPMD mode
# run like this:
#    mpirun -n 4 python ./covariance_spmd.py

import daal4py as d4p
from numpy import loadtxt


if __name__ == "__main__":
    # Initialize SPMD mode
    d4p.daalinit()

    # Each process gets its own data
    infile = "./data/distributed/covcormoments_dense_" + str(d4p.my_procid()+1) + ".csv"
    data = loadtxt(infile, delimiter=',')

    # Create algorithm with distributed mode
    alg = d4p.covariance(method="defaultDense", distributed=True)

    # Perform computation
    res = alg.compute(data)

    # covariance result objects provide correlation, covariance and mean
    assert res.covariance.shape == (data.shape[1], data.shape[1])
    assert res.mean.shape == (1, data.shape[1])
    assert res.correlation.shape == (data.shape[1], data.shape[1])

    # Print message to show that job is done
    # To see more details look at batch example
    print('All looks good!')
    d4p.daalfini()
