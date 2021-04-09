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

#ifdef _WIN32
    #define NOMINMAX
#endif
#include <string>
#include <cinttypes>
#include "oneapi/dal/common.hpp"

#include <numpy/arrayobject.h>

#define ONEDAL_2021_3_VERSION (2021 * 10000 + 3 * 100)

static std::string to_std_string(PyObject * o)
{
    return PyUnicode_AsUTF8(o);
}

#define SET_NPY_FEATURE(_T, _FUNCT, _EXCEPTION) \
    switch (_T)                                 \
    {                                           \
    case NPY_DOUBLE:                            \
    case NPY_CDOUBLE:                           \
    case NPY_DOUBLELTR:                         \
    case NPY_CDOUBLELTR:                        \
    {                                           \
        _FUNCT(double);                         \
        break;                                  \
    }                                           \
    case NPY_FLOAT:                             \
    case NPY_CFLOAT:                            \
    case NPY_FLOATLTR:                          \
    case NPY_CFLOATLTR:                         \
    {                                           \
        _FUNCT(float);                          \
        break;                                  \
    }                                           \
    case NPY_INT32:                             \
    {                                           \
        _FUNCT(std::int32_t);                   \
        break;                                  \
    }                                           \
    case NPY_UINT32:                            \
    {                                           \
        _FUNCT(std::uint32_t);                  \
        break;                                  \
    }                                           \
    case NPY_INT64:                             \
    {                                           \
        _FUNCT(std::int64_t);                   \
        break;                                  \
    }                                           \
    case NPY_UINT64:                            \
    {                                           \
        _FUNCT(std::uint64_t);                  \
        break;                                  \
    }                                           \
    default: _EXCEPTION;                        \
    };

#define SET_CTYPE_NPY_FROM_DAL_TYPE(_T, _FUNCT, _EXCEPTION) \
    switch (_T)                                             \
    {                                                       \
    case dal::data_type::float32:                           \
    {                                                       \
        _FUNCT(float, NPY_FLOAT32);                         \
        break;                                              \
    }                                                       \
    case dal::data_type::float64:                           \
    {                                                       \
        _FUNCT(double, NPY_FLOAT64);                        \
        break;                                              \
    }                                                       \
    case dal::data_type::int32:                             \
    {                                                       \
        _FUNCT(std::int32_t, NPY_INT32);                    \
        break;                                              \
    }                                                       \
    default: _EXCEPTION;                                    \
    };
