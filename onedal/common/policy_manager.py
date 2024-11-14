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


class Policy:
    """Encapsulates backend policies for a unified interface with auxiliary information"""

    def __init__(self, policy_module, queue, is_dpc, is_spmd):
        self.policy = policy_module(queue) if queue else policy_module()
        self.is_dpc = is_dpc
        self.is_spmd = is_spmd
        if is_dpc:
            if queue is None:
                raise ValueError("DPC++ policy requires a queue")
            self._queue = queue

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def __repr__(self) -> str:
        return f"Policy({self.policy}, is_dpc={self.is_dpc}, is_spmd={self.is_spmd})"


class PolicyManager:
    def __init__(self, backend):
        self.backend = backend

    @staticmethod
    def get_queue(*data):
        if not data:
            return
        if iface := getattr(data[0], "__sycl_usm_array_interface__", None):
            queue = iface.get("syclobj")
            if not queue:
                raise KeyError("No syclobj in provided data")
            return queue

    def get_policy(self, provided_queue, *data):
        data_queue = PolicyManager.get_queue(*data)
        queue = provided_queue if provided_queue is not None else data_queue

        if not self.backend.is_dpc and queue is not None:
            raise RuntimeError("Operations using queues require the DPC backend")

        if self.backend.is_spmd and queue is not None:
            backend_policy = self.backend.spmd_data_parallel_policy
            is_dpc = True
            is_spmd = True
        elif self.backend.is_dpc and queue is not None:
            backend_policy = self.backend.data_parallel_policy
            is_dpc = True
            is_spmd = False
        else:
            backend_policy = self.backend.host_policy
            is_dpc = False
            is_spmd = False
        return Policy(backend_policy, queue, is_dpc, is_spmd)
