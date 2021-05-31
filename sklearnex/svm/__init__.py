#!/usr/bin/env python
#===============================================================================
# Copyright 2021 Intel Corporation
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
#===============================================================================


from .._utils import get_sklearnex_version
if get_sklearnex_version((2021, 'P', 300)):
    from .svr import SVR
    from .svc import SVC
    from .nusvr import NuSVR
    from .nusvc import NuSVC
    __all__ = ['SVR', 'SVC', 'NuSVC', 'NuSVR']
else:
    from daal4py.sklearn.svm import SVC
    __all__ = ['SVC']
