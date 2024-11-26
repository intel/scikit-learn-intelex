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
from typing import Any, Callable, Literal, Optional

from onedal import Backend, _default_backend, _spmd_backend
from onedal._device_offload import SyclQueueManager

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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Dispatch to backend function with the appropriate policy which is determined from the global queue"""
        if not args and not kwargs:
            # immediate dispatching without any arguments, in particular no policy
            return self.method()

        # use globally configured queue (from `target_offload` configuration or provided data)
        queue = getattr(SyclQueueManager.get_global_queue(), "implementation", None)

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
        try:
            return self.method(policy, *args, **kwargs)
        except:
            raise RuntimeError(
                f"Error in dispatching to backend function for device {policy.get_device_id()} ({policy.get_device_name()})"
            )

    def __repr__(self) -> str:
        return f"BackendFunction({self.backend}.{self.name})"


def __decorator(
    method: Callable[..., Any],
    backend_manager: BackendManager,
    module_name: str,
    lookup_name: Optional[str],
) -> Callable[..., Any]:
    """Decorator to bind a method to the specified backend"""
    if lookup_name is None:
        lookup_name = method.__name__

    if backend_manager.get_backend_type() == "none":
        raise RuntimeError("Internal __decorator() should not be called with no backend")

    backend_method = backend_manager.get_backend_component(module_name, lookup_name)
    wrapped_method = BackendFunction(
        backend_method,
        backend_manager.backend,
        name=f"{module_name}.{method.__name__}",
    )

    backend_type = backend_manager.get_backend_type()
    logger.debug(
        f"Assigned method '<{backend_type}_backend>.{module_name}.{lookup_name}' to '{method.__qualname__}'"
    )

    return wrapped_method


def bind_default_backend(module_name: str, lookup_name: Optional[str] = None):
    def decorator(method: Callable[..., Any]):
        # grab the lookup_name from outer scope
        nonlocal lookup_name

        if _default_backend is None:
            logger.debug(
                f"Default backend unavailable, skipping decoration for '{method.__name__}'"
            )
            return method

        return __decorator(method, default_manager, module_name, lookup_name)

    return decorator


def bind_spmd_backend(module_name: str, lookup_name: Optional[str] = None):
    def decorator(method: Callable[..., Any]):
        # grab the lookup_name from outer scope
        nonlocal lookup_name

        if _spmd_backend is None:
            logger.debug(
                f"SPMD backend unavailable, skipping decoration for '{method.__name__}'"
            )
            return method

        __decorator(method, spmd_manager, module_name, lookup_name)

    return decorator
