#!/usr/bin/env python
#===============================================================================
# Copyright 2021 Intel Corporation
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

from daal4py.sklearn._utils import daal_check_version

if daal_check_version((2023, 'P', 200)):
    from onedal.kmeans import KMeans as onedal_KMeans
    from sklearn.cluster import KMeans as sklearn_KMeans

    def _validate_center_shape(X, n_centers, centers):
        """Check if centers is compatible with X and n_centers"""
        if centers.shape[0] != n_centers:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of clusters {n_centers}.")
        if centers.shape[1] != X.shape[1]:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of features of the data {X.shape[1]}.")


    def _tolerance(X, rtol):
        """Compute absolute tolerance from the relative tolerance"""
        if rtol == 0.0:
            return rtol
        if sp.issparse(X):
            variances = mean_variance_axis(X, axis=0)[1]
            mean_var = np.mean(variances)
        else:
            mean_var = np.var(X, axis=0).mean()
        return mean_var * rtol

else:
    from daal4py.sklearn.cluster import KMeans
