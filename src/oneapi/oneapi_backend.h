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

#ifndef __ONEAPI_BACKEND_H_INCLUDED__
#define __ONEAPI_BACKEND_H_INCLUDED__

#include "daal_sycl.h"
#ifndef DAAL_SYCL_INTERFACE
    #include <type_traits>
    #include <memory>
static_assert(false, "DAAL_SYCL_INTERFACE not defined")
#endif

#ifdef _WIN32
#define _ONEAPI_BACKEND_EXPORT __declspec(dllexport)
#else
#define _ONEAPI_BACKEND_EXPORT
#endif

class _ONEAPI_BACKEND_EXPORT PySyclExecutionContext
{
public:
    // Construct from given device provided as string
    PySyclExecutionContext(const std::string & dev);
    ~PySyclExecutionContext();

    void apply();

private:
    daal::services::SyclExecutionContext * m_ctxt;
};

template <typename T>
_ONEAPI_BACKEND_EXPORT void* to_device(T * ptr, int * shape);

template <typename T>
_ONEAPI_BACKEND_EXPORT void delete_device_data(void * ptr);

// take a sycl buffer and convert ti oneDAL NT
template <typename T, bool is_device_data>
_ONEAPI_BACKEND_EXPORT daal::data_management::NumericTablePtr * to_daal_nt(void * ptr, int * shape);

// return a device data from a SyclHomogenNumericTable
template <typename T>
_ONEAPI_BACKEND_EXPORT void * fromdaalnt(daal::data_management::NumericTablePtr * ptr);

#endif // __ONEAPI_BACKEND_H_INCLUDED__
