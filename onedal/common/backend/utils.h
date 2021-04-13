/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <numpy/arrayobject.h>

#define ONEDAL_2021_3_VERSION (2021 * 10000 + 3 * 100)

#ifdef _WIN32
    #define NOMINMAX
    #define ONEDAL_BACKEND_EXPORT __declspec(dllexport)
#else
    #define ONEDAL_BACKEND_EXPORT
#endif

namespace oneapi::dal::python
{
class thread_allow
{
public:
    thread_allow() { allow(); }
    ~thread_allow() { disallow(); }
    void allow() { _save = PyEval_SaveThread(); }
    void disallow()
    {
        if (_save)
        {
            PyEval_RestoreThread(_save);
            _save = NULL;
        }
    }

private:
    PyThreadState * _save;
};

} // namespace oneapi::dal::python
