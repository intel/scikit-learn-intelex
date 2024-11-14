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

from unittest.mock import MagicMock

import pytest

from onedal.common.policy_manager import PolicyManager


# Define a simple backend module for testing
class DummyBackend:
    def __init__(self, is_dpc, is_spmd):
        self.is_dpc = is_dpc
        self.is_spmd = is_spmd

    def data_parallel_policy(self, queue):
        return f"data_parallel_policy({queue})"

    def spmd_data_parallel_policy(self, queue):
        return f"spmd_data_parallel_policy({queue})"

    def host_policy(self):
        return "host_policy"


@pytest.fixture
def backend_dpc():
    return DummyBackend(is_dpc=True, is_spmd=False)


@pytest.fixture
def backend_spmd():
    return DummyBackend(is_dpc=True, is_spmd=True)


@pytest.fixture
def backend_host():
    return DummyBackend(is_dpc=False, is_spmd=False)


@pytest.fixture
def policy_manager_dpc(backend_dpc):
    return PolicyManager(backend_dpc)


@pytest.fixture
def policy_manager_spmd(backend_spmd):
    return PolicyManager(backend_spmd)


@pytest.fixture
def policy_manager_host(backend_host):
    return PolicyManager(backend_host)


def test_get_queue_with_sycl_usm_array_interface():
    data = [MagicMock()]
    data[0].__sycl_usm_array_interface__ = {"syclobj": "queue"}
    queue = PolicyManager.get_queue(*data)
    assert queue == "queue"


def test_get_queue_without_sycl_usm_array_interface():
    data = [MagicMock()]
    queue = PolicyManager.get_queue(*data)
    assert queue is None


def test_get_policy_with_provided_queue(policy_manager_dpc):
    provided_queue = MagicMock()
    policy = policy_manager_dpc.get_policy(provided_queue)
    assert policy.policy == "data_parallel_policy({})".format(provided_queue)
    assert policy.is_dpc is True
    assert policy.is_spmd is False


def test_get_policy_with_data_queue(policy_manager_dpc):
    data = [MagicMock()]
    data[0].__sycl_usm_array_interface__ = {"syclobj": MagicMock()}
    policy = policy_manager_dpc.get_policy(None, *data)
    assert policy.policy == "data_parallel_policy({})".format(
        data[0].__sycl_usm_array_interface__["syclobj"]
    )
    assert policy.is_dpc is True
    assert policy.is_spmd is False


def test_get_policy_with_host_backend_and_queue(policy_manager_host):
    provided_queue = MagicMock()
    with pytest.raises(
        RuntimeError, match="Operations using queues require the DPC backend"
    ):
        policy_manager_host.get_policy(provided_queue)


def test_get_policy_with_host_backend(policy_manager_host):
    policy = policy_manager_host.get_policy(None)
    assert policy.policy == "host_policy"
    assert policy.is_dpc is False
    assert policy.is_spmd is False


def test_get_policy_with_dpc_backend_no_queue(policy_manager_dpc):
    policy = policy_manager_dpc.get_policy(None)
    assert policy.policy == "host_policy"
    assert policy.is_dpc is False
    assert policy.is_spmd is False


def test_get_policy_with_spmd_backend_and_queue(policy_manager_spmd):
    provided_queue = MagicMock()
    policy = policy_manager_spmd.get_policy(provided_queue)
    assert policy.policy == "spmd_data_parallel_policy({})".format(provided_queue)
    assert policy.is_dpc is True
    assert policy.is_spmd is True


def test_get_policy_with_spmd_backend_no_queue(policy_manager_spmd):
    policy = policy_manager_spmd.get_policy(None)
    assert policy.policy == "host_policy"
    assert policy.is_dpc is False
    assert policy.is_spmd is False
