# ===============================================================================
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
# ===============================================================================

from daal4py.sklearn.cluster import KMeans
from onedal._device_offload import support_usm_ndarray

# Note: `sklearnex.cluster.KMeans` only has functional
# sycl GPU support. No GPU device will be offloaded.
KMeans.fit = support_usm_ndarray(queue_param=False)(KMeans.fit)
KMeans.fit_predict = support_usm_ndarray(queue_param=False)(KMeans.fit_predict)
KMeans.predict = support_usm_ndarray(queue_param=False)(KMeans.predict)
KMeans.score = support_usm_ndarray(queue_param=False)(KMeans.score)
