# ==============================================================================
# Copyright 2023 Intel Corporation
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

from sklearnex import config_context

try:
    import dpctl

    dpctl_is_available = True
    gpu_is_available = dpctl.has_gpu_devices()
except (ImportError, ModuleNotFoundError):
    dpctl_is_available = False


@pytest.mark.skipif(
    not dpctl_is_available or gpu_is_available,
    reason="GPU device should not be available for this test "
    "to see raised 'SyclQueueCreationError'. "
    "'dpctl' module is required for test.",
)
def test_config_context_in_parallel(with_sklearnex):
    from sklearn.datasets import make_classification
    from sklearn.ensemble import BaggingClassifier
    from sklearn.svm import SVC

    x, y = make_classification(random_state=42)
    try:
        with config_context(target_offload="gpu", allow_fallback_to_host=False):
            BaggingClassifier(SVC(), n_jobs=2).fit(x, y)
        raise ValueError(
            "'SyclQueueCreationError' wasn't raised " "for non-existing 'gpu' device"
        )
    except dpctl._sycl_queue.SyclQueueCreationError:
        pass
