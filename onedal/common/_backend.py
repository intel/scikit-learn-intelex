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

logger = logging.getLogger(__name__)

# define types for backend functions: default, dpc, spmd
BackendType = Literal["none", "host", "dpc", "spmd"]

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class BackendManager:
    def __init__(self, backend_module):
        self.backend = backend_module

    def get_backend_type(self) -> BackendType:
        if self.backend is None:
            return "none"
        if self.backend.is_spmd:
            return "spmd"
        if self.backend.is_dpc:
            return "dpc"
        return "host"

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


default_manager = BackendManager(_default_backend)
spmd_manager = BackendManager(_spmd_backend)


class BackendFunction:
    """Wrapper around backend function to allow setting auxiliary information"""

    def __init__(
        self,
        method: Callable[..., Any],
        backend: Backend,
        name: str,
        no_policy: bool,
    ):
        self.method = method
        self.name = name
        self.backend = backend
        self.no_policy = no_policy

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Dispatch to backend function with the appropriate policy which is determined from the global queue"""
        if not args and not kwargs:
            # immediate dispatching without any arguments, in particular no policy
            return self.method()

        if self.no_policy:
            return self.method(*args, **kwargs)

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


def __decorator(
    method: Callable[..., Any],
    backend_manager: BackendManager,
    module_name: str,
    lookup_name: Optional[str],
    no_policy: bool,
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
        no_policy=no_policy,
    )

    backend_type = backend_manager.get_backend_type()
    logger.debug(
        f"Assigned method '<{backend_type}_backend>.{module_name}.{lookup_name}' to '{method.__qualname__}'"
    )

    return wrapped_method


def bind_default_backend(
    module_name: str, lookup_name: Optional[str] = None, no_policy=False
):
    """
    Decorator to bind a method from the default backend to a class.

    This decorator binds a method implementation from the default backend (host/dpc).
    If the default backend is unavailable, the method is returned without modification.

    Parameters:
    ----------
    module_name : str
        The name of the module where the target function is located (e.g. `covariance`).
    lookup_name : Optional[str], optional
        The name of the method to look up in the backend module. If not provided,
        the name of the decorated method is used.
    no_policy : bool, optional
        If True, the method will be decorated without a policy. Default is False.

    Returns:
    -------
    Callable[..., Any]
        The decorated method bound to the implementation in default backend, or the original
        method if the default backend is unavailable.
    """

    def decorator(method: Callable[..., Any]):
        # grab the lookup_name from outer scope
        nonlocal lookup_name

        if _default_backend is None:
            logger.debug(
                f"Default backend unavailable, skipping decoration for '{method.__name__}'"
            )
            return method

        return __decorator(method, default_manager, module_name, lookup_name, no_policy)

    return decorator


def bind_spmd_backend(
    module_name: str, lookup_name: Optional[str] = None, no_policy=False
):
    """
    Decorator to bind a method from the SPMD backend to a class.

    This decorator binds a method implementation from the SPMD backend.
    If the SPMD backend is unavailable, the method is returned without modification.

    Parameters:
    ----------
    module_name : str
        The name of the module where the target function is located (e.g. `covariance`).
    lookup_name : Optional[str], optional
        The name of the method to look up in the backend module. If not provided,
        the name of the decorated method is used.
    no_policy : bool, optional
        If True, the method will be decorated without a policy. Default is False.

    Returns:
    -------
    Callable[..., Any]
        The decorated method bound to the implementation in SPMD backend, or the original
        method if the SPMD backend is unavailable.
    """

    def decorator(method: Callable[..., Any]):
        # grab the lookup_name from outer scope
        nonlocal lookup_name

        if _spmd_backend is None:
            logger.debug(
                f"SPMD backend unavailable, skipping decoration for '{method.__name__}'"
            )
            return method

        __decorator(method, spmd_manager, module_name, lookup_name, no_policy)

    return decorator
