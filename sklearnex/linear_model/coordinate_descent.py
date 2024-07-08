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

from daal4py.sklearn.linear_model import ElasticNet, Lasso
from onedal._device_offload import support_usm_ndarray

# Note: `sklearnex.linear_model.ElasticNet` only has functional
# sycl GPU support. No GPU device will be offloaded.
ElasticNet.fit = support_usm_ndarray(queue_param=False)(ElasticNet.fit)
ElasticNet.predict = support_usm_ndarray(queue_param=False)(ElasticNet.predict)
ElasticNet.score = support_usm_ndarray(queue_param=False)(ElasticNet.score)

# Note: `sklearnex.linear_model.Lasso` only has functional
# sycl GPU support. No GPU device will be offloaded.
Lasso.fit = support_usm_ndarray(queue_param=False)(Lasso.fit)
Lasso.predict = support_usm_ndarray(queue_param=False)(Lasso.predict)
Lasso.score = support_usm_ndarray(queue_param=False)(Lasso.score)
