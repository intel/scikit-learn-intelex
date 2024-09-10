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

"""Check availability of DPPY imports in one place"""


def is_dpctl_available(version=""):
    """Checks availability of DPCtl package"""
    try:
        import dpctl
        import dpctl.tensor as dpt

        dpctl_available = True
    except ImportError:
        dpctl_available = False
    if dpctl_available and not version == "":
        dpctl_available = dpctl.__version__ >= version
    return dpctl_available


def is_dpnp_available(version=""):
    """Checks availability of DPNP package"""
    try:
        import dpnp

        dpnp_available = True
    except ImportError:
        dpnp_available = False
    if dpnp_available and not version == "":
        dpnp_available = dpnp.__version__ >= version
    return dpnp_available


dpctl_available = is_dpctl_available()
dpctl_available = is_dpnp_available()
