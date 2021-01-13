#!/usr/bin/env python
#===============================================================================
# Copyright 2014-2021 Intel Corporation
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

try:
    from _daal4py import *
    from _daal4py import _get__version__, _get__daal_link_version__, _get__daal_run_version__, __has_dist__
except ImportError as e:
    s = str(e)
    if 'libfabric' in s:
        raise ImportError(s + '\n\nActivating your conda environment or sourcing mpivars.[c]sh/psxevars.[c]sh may solve the issue.\n')
    raise

import logging
import warnings
import os
import sys
logLevel = os.environ.get("IDP_SKLEARN_VERBOSE")
try:
    if not logLevel is None:
        logging.basicConfig(stream=sys.stdout, format='%(levelname)s: %(message)s', level=logLevel.upper())
except:
    warnings.warn('Unknown level "{}" for logging.\n'
                    'Please, use one of "CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG".'.format(logLevel))
