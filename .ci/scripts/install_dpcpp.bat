REM SPDX-FileCopyrightText: 2023 Intel Corporation
REM
REM SPDX-License-Identifier: MIT

curl.exe --output dpcpp_compiler_installer.exe --url https://registrationcenter-download.intel.com/akdlm/IRC_NAS/d0c91df9-1613-4edd-bd62-ea982255a13a/w_dpcpp-cpp-compiler_p_2023.2.0.49257_offline.exe --retry 5 --retry-delay 5
dpcpp_compiler_installer.exe -s -a --silent --eula accept
