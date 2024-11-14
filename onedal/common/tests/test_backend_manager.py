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


import pytest

from onedal.common.backend_manager import BackendManager


# Define a simple backend module for testing
class DummyBackend:
    class Module:
        class Submodule:
            def method(self, *args, **kwargs):
                return "method_result"

        def __init__(self):
            self.submodule_instance = self.Submodule()

        def method(self, *args, **kwargs):
            return "method_result"

    def __init__(self):
        self.module_instance = self.Module()

    @property
    def module(self):
        return self.module_instance


@pytest.fixture
def backend_manager():
    backend = DummyBackend()
    return BackendManager(backend)


def test_get_backend_component_with_method(backend_manager):
    result = backend_manager.get_backend_component("module", "method")
    assert result() == "method_result"


def test_get_backend_component_with_submodule_method(backend_manager):
    result = backend_manager.get_backend_component("module.submodule_instance", "method")
    assert result() == "method_result"


def test_get_backend_component_with_invalid_module(backend_manager):
    with pytest.raises(AttributeError):
        backend_manager.get_backend_component("invalid_module", "method")


def test_get_backend_component_with_invalid_submodule(backend_manager):
    with pytest.raises(AttributeError):
        backend_manager.get_backend_component("module.invalid_submodule", "method")


def test_get_backend_component_with_invalid_method(backend_manager):
    with pytest.raises(AttributeError):
        backend_manager.get_backend_component(
            "module", "submodule_instance.invalid_method"
        )


def test_get_backend_component_with_multiple_methods(backend_manager):
    class ExtendedDummyBackend(DummyBackend):
        class Module(DummyBackend.Module):
            class Submodule(DummyBackend.Module.Submodule):
                def another_method(self, *args, **kwargs):
                    return "another_method_result"

            def __init__(self):
                super().__init__()
                self.submodule_instance = self.Submodule()

        def __init__(self):
            self.module_instance = self.Module()

    backend_manager.backend = ExtendedDummyBackend()
    result = backend_manager.get_backend_component(
        "module.submodule_instance", "another_method"
    )
    assert result() == "another_method_result"


def test_get_backend_component_with_deeply_nested_submodules(backend_manager):
    class DeeplyNestedDummyBackend(DummyBackend):
        class Module(DummyBackend.Module):
            class Submodule(DummyBackend.Module.Submodule):
                class DeepSubmodule:
                    def deep_method(self, *args, **kwargs):
                        return "deep_method_result"

                def __init__(self):
                    super().__init__()
                    self.deep_submodule_instance = self.DeepSubmodule()

            def __init__(self):
                super().__init__()
                self.submodule_instance = self.Submodule()

        def __init__(self):
            self.module_instance = self.Module()

    backend_manager.backend = DeeplyNestedDummyBackend()
    result = backend_manager.get_backend_component(
        "module.submodule_instance.deep_submodule_instance", "deep_method"
    )
    assert result() == "deep_method_result"
