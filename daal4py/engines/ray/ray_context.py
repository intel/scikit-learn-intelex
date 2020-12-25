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

import ray
from daal4py.engines.context import Context

class RayContext(Context):

    def current_node_id(self):
        return ray.services.get_node_ip_address()

    def node_ids(self):
        return ray.state.node_ids()

    def available_resources(self):
        return ray.available_resources()

    def get_world_size(self):
        return len(ray.nodes())

    def cluster_resources(self):
        return ray.cluster_resources()
