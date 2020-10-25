REM SPDX-FileCopyrightText: 2020 Intel Corporation
REM
REM SPDX-License-Identifier: MIT

set URL=%1
set COMPONENTS=%2

curl.exe --output webimage.exe --url %URL%
start /b /wait webimage.exe -s -x -f webimage_extracted 
webimage_extracted\bootstrapper.exe -s --action install --components=%COMPONENTS% --eula=accept --continue-with-optional-error=yes -p=NEED_VS2017_INTEGRATION=0 -p=NEED_VS2019_INTEGRATION=0
