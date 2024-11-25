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

import logging
from contextlib import contextmanager
from types import MethodType
from typing import Any, Callable, Literal, Optional

from onedal import Backend, _default_backend, _spmd_backend
from onedal._device_offload import SyclQueueManager
from onedal.common.policy_manager import PolicyManager

from .backend_manager import BackendManager

logger = logging.getLogger(__name__)

default_manager = BackendManager(_default_backend)
spmd_manager = BackendManager(_spmd_backend)

# define types for backend functions: default, dpc, spmd
BackendType = Literal["host", "dpc", "spmd"]

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class BackendFunction:
    """Wrapper around backend function to allow setting auxiliary information"""

    def __init__(
        self,
        method: Callable[..., Any],
        backend: Backend,
        name: str,
    ):
        self.method = method
        self.name = name
        self.backend = backend

    def __call__(self, *args: Any, queue=None, **kwargs: Any) -> Any:
        """Dispatch to backend function with the appropriate policy which is determined from the provided or global queue"""
        if not args:
            # immediate dispatching without args, i.e. without data
            return self.method(**kwargs)

        if queue is None:
            # use globally configured queue (from `target_offload` configuration or provided data)
            queue = SyclQueueManager.get_global_queue()

        if queue is not None and not (self.backend.is_dpc or self.backend.is_spmd):
            raise RuntimeError("Operations using queues require the DPC/SPMD backend")

        # craft the correct policy including the device queue
        if queue is None:
            policy = self.backend.host_policy()
        elif self.backend.is_spmd:
            policy = self.backend.spmd_data_parallel_policy(queue)
        elif self.backend.is_dpc:
            policy = self.backend.data_parallel_policy(queue)
        else:
            policy = self.backend.host_policy()

        # dispatch to backend function
        return self.method(policy, *args, **kwargs)

    def __repr__(self) -> str:
        return f"BackendFunction({self.backend}.{self.name})"


def bind_default_backend(module_name: str, lookup_name: Optional[str] = None):
    def decorator(method: Callable[..., Any]):
        # grab the lookup_name from outer scope
        nonlocal lookup_name

        if lookup_name is None:
            lookup_name = method.__name__

        if _default_backend is None:
            logger.debug(
                f"Default backend unavailable, skipping decoration for '{method.__name__}'"
            )
            return method

        backend_method = default_manager.get_backend_component(module_name, lookup_name)
        wrapped_method = BackendFunction(
            backend_method,
            _default_backend,
            name=f"{module_name}.{method.__name__}",
        )

        backend_name = "dpc" if _default_backend.is_dpc else "host"
        logger.debug(
            f"Assigned method '<{backend_name}_backend>.{module_name}.{lookup_name}' to '{method.__qualname__}'"
        )

        return wrapped_method

    return decorator


def bind_spmd_backend(module_name: str, lookup_name: Optional[str] = None):
    def decorator(method: Callable[..., Any]):
        # grab the lookup_name from outer scope
        nonlocal lookup_name

        if lookup_name is None:
            lookup_name = method.__name__

        if _spmd_backend is None:
            logger.debug(
                f"SPMD backend unavailable, skipping decoration for '{method.__name__}'"
            )
            return method

        backend_method = spmd_manager.get_backend_component(module_name, lookup_name)
        wrapped_method = BackendFunction(
            backend_method,
            _spmd_backend,
            name=f"{module_name}.{method.__name__}",
        )

        logger.debug(
            f"Assigned method '<spmd_backend>.{module_name}.{lookup_name}' to '{method.__qualname__}' "
        )

        return wrapped_method

    return decorator
