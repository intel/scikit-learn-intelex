#
#*******************************************************************************
# Copyright 2014-2020 Intel Corporation
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
#******************************************************************************/

import numpy as np

def getFPType(X):
    dt = getattr(X, 'dtype', None)
    if dt == np.double:
        return "double"
    elif dt == np.single:
        return "float"
    else:
        raise ValueError("Input array has unexpected dtype = {}".format(dt))

def make2d(X):
    if np.isscalar(X):
        X = np.asarray(X)[np.newaxis, np.newaxis]
    elif isinstance(X, np.ndarray) and X.ndim == 1:
        X = X.reshape((X.size, 1))
    return X

method_uses_daal = "uses Intel® DAAL solver"
method_uses_sklearn = "uses original Scikit-learn solver"
method_uses_sklearn_arter_daal = "uses original Scikit-learn solver, because the task was not solved with Intel® DAAL"
