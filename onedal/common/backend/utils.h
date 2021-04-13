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

#include "oneapi/dal/common.hpp"
#include <numpy/arrayobject.h>
#include <string>

#define ONEDAL_2021_3_VERSION (2021 * 10000 + 3 * 100)

namespace oneapi::dal::python
{
static std::string to_std_string(PyObject * o)
{
    return PyUnicode_AsUTF8(o);
}

class thread_allow
{
public:
    thread_allow() { allow(); }

    ~thread_allow() { disallow(); }

private:
    void allow() { save_ = PyEval_SaveThread(); }

    void disallow()
    {
        if (save_)
        {
            PyEval_RestoreThread(save_);
            save_ = NULL;
        }
    }

    PyThreadState * save_;
};

} // namespace oneapi::dal::python
