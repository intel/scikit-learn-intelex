/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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

#ifndef __ONEAPI_H_INCLUDED__
#define __ONEAPI_H_INCLUDED__

#include "oneapi_backend.h"
#include "numpy/ndarraytypes.h"
#include "oneapi_api.h"

static void * to_device(void * ptr, int typ, int * shape)
{
    switch (typ)
    {
    case NPY_DOUBLE: return to_device(reinterpret_cast<double *>(ptr), shape); break;
    case NPY_FLOAT: return to_device(reinterpret_cast<float *>(ptr), shape); break;
    case NPY_INT: return to_device(reinterpret_cast<int *>(ptr), shape); break;
    default: throw std::invalid_argument("invalid input array type (must be double, float or int)");
    }
}

template <bool is_device_data>
inline void * to_daal_nt(void * ptr, int typ, int * shape)
{
    switch (typ)
    {
    case NPY_DOUBLE: return to_daal_nt<double, is_device_data>(ptr, shape); break;
    case NPY_FLOAT: return to_daal_nt<float, is_device_data>(ptr, shape); break;
    case NPY_INT: return to_daal_nt<int, is_device_data>(ptr, shape); break;
    default: throw std::invalid_argument("invalid input array type (must be double, float or int)");
    }
}

static void * to_daal_sycl_nt(void * ptr, int typ, int * shape)
{
    return to_daal_nt<true>(ptr, typ, shape);
}

static void * to_daal_host_nt(void * ptr, int typ, int * shape)
{
    return to_daal_nt<false>(ptr, typ, shape);
}

static void delete_device_data(void * ptr, int typ)
{
    if (ptr == nullptr)
        return;

    switch (typ)
    {
    case NPY_DOUBLE: delete_device_data<double>(ptr); break;
    case NPY_FLOAT: delete_device_data<float>(ptr); break;
    case NPY_INT: delete_device_data<int>(ptr); break;
    default: throw std::invalid_argument("invalid array type (must be double, float or int)");
    }
}

static std::string to_std_string(PyObject * o)
{
    return PyUnicode_AsUTF8(o);
}

void * c_make_py_from_sycltable(void * _ptr, int typ)
{
    auto ptr = reinterpret_cast<daal::data_management::NumericTablePtr *>(_ptr);

    switch (typ)
    {
    case NPY_DOUBLE: return fromdaalnt<double>(ptr); break;
    case NPY_FLOAT: return fromdaalnt<float>(ptr); break;
    case NPY_INT: return fromdaalnt<int>(ptr); break;
    default: throw std::invalid_argument("invalid output array type (must be double, float or int)");
    }
    return NULL;
}

#endif // __ONEAPI_H_INCLUDED__
