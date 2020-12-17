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

import os

class PyOneCCL:
    def __init__(self, world_size, master_node_ip, port="51234"):
        self.ccl_root = os.getenv("CCL_ROOT")
        self.i_mpi_root = os.getenv("I_MPI_ROOT")
        self.world_size = world_size
        self.ccl_kvs_ip_port = master_node_ip + "_" + port

    def set_env(self):
        os.environ["CCL_PM_TYPE"] = "resizable"
        os.environ["CCL_ATL_TRANSPORT"] = "ofi"
        os.environ["CCL_KVS_IP_EXCHANGE"] = "env"
        os.environ["CCL_WORLD_SIZE"] = str(self.world_size)
        os.environ["CCL_KVS_IP_PORT"] = self.ccl_kvs_ip_port
        os.environ[
            "CCL_ROOT"] = self.ccl_root
        os.environ[
            "I_MPI_ROOT"] = self.i_mpi_root
