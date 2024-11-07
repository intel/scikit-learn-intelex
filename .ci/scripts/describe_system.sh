#!/bin/bash
#===============================================================================
# Copyright 2022 Intel Corporation
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

# CPU info
echo "CPU features:"
if [ -x "$(command -v lscpu)" ]; then
    lscpu
elif [ "$(pip list | grep cpufeature)" != "" ]; then
    python -c "import cpufeature; cpufeature.print_features()"
else
    echo "Unable to get CPU features via lscpu or python/cpufeature"
fi
echo
# OS info
echo "Operating system:"
if [ -x "$(command -v uname)" ]; then
    uname -a
elif [ -x "$(command -v python)" ]; then
    python -c "import platform; print(platform.platform())"
else
    echo "Unable to get operating system via uname or python/platform"
fi
echo
# meminfo
if [ -f /proc/meminfo ]; then
    cat /proc/meminfo
fi
# Compilers
echo "Compilers:"
if [ -x "$(command -v gcc)" ]; then
    echo "GNU:"
    gcc --version
fi
if [ -x "$(command -v clang)" ]; then
    echo "Clang:"
    clang --version
fi
if [ -x "$(command -v icx)" ]; then
    echo "ICX:"
    icx --version
fi
if [ -x "$(command -v icpx)" ]; then
    echo "ICPX:"
    icpx --version
fi
echo
# SYCL devices
if [ -x "$(command -v sycl-ls)" ]; then
    sycl-ls
fi
