# ==============================================================================
# Copyright 2024 Intel Corporation
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
# ==============================================================================

from ...common._backend import DefaultPolicyOverride, bind_spmd_backend
from ...decomposition import IncrementalPCA as base_IncrementalPCA


class IncrementalPCA(base_IncrementalPCA):
    """
    Distributed incremental estimator for PCA based on oneDAL implementation.
    Allows for distributed PCA computation if data is split into batches.

    API is the same as for `onedal.decomposition.IncrementalPCA`
    """

    @bind_spmd_backend("decomposition")
    def _get_policy(self, queue, *data): ...

    @bind_spmd_backend("decomposition.dim_reduction")
    def finalize_train(self, policy, params, partial_result): ...

    def partial_fit(self, X, queue):
        # partial fit performed by parent backend, therefore default policy required
        with DefaultPolicyOverride(self):
            return super().partial_fit(X, queue)

    def infer(self, policy, params, X, model):
        # infer runs in parent backend, therefore default policy required
        with DefaultPolicyOverride(self):
            return super().infer(policy, params, X, model)
