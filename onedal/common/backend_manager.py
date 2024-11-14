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


class BackendManager:
    def __init__(self, backend_module):
        self.backend = backend_module

    def get_backend_component(self, module_name: str, component_name: str):
        """Get a component of the backend module.

        Args:
            module(str): The module to get the component from.
            component: The component to get from the module.

        Returns:
            The component of the module.
        """
        submodules = module_name.split(".")
        module = getattr(self.backend, submodules[0])
        for submodule in submodules[1:]:
            module = getattr(module, submodule)

        # component can be provided like submodule.method, there can be arbitrary number of submodules
        # and methods
        result = module
        for part in component_name.split("."):
            result = getattr(result, part)

        return result
