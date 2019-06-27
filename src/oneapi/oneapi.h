/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
/******************************************************************************/

#ifndef __ONEAPI_H_INCLUDED__
#define __ONEAPI_H_INCLUDED__

#include "daal_sycl.h"

#ifndef DAAL_SYCL_INTERFACE
#include <type_traits>
static_assert(false, "DAAL_SYCL_INTERFACE not defined")
#endif

// Wrapping DAAL's SyclExecutionContext
// At construction time we optionally provide the device selector or a queue
class PySyclExecutionContext
{
public:
    // Construct from given device selector
    PySyclExecutionContext(const cl::sycl::device_selector & dev_sel = cl::sycl::default_selector())
        : m_ctxt(new daal::services::SyclExecutionContext(cl::sycl::queue(dev_sel)))
    {}
    // Construct from given queue (implicitly linked to device)
    PySyclExecutionContext(const cl::sycl::queue & q)
        : m_ctxt(new daal::services::SyclExecutionContext(q))
    {}
    // Construct from given device provided as string
    PySyclExecutionContext(const std::string & dev)
        : m_ctxt(NULL)
    {
        if(dev == "gpu") m_ctxt = new daal::services::SyclExecutionContext(cl::sycl::queue(cl::sycl::gpu_selector()));
        else if(dev == "cpu") m_ctxt = new daal::services::SyclExecutionContext(cl::sycl::queue(cl::sycl::cpu_selector()));
        else m_ctxt = new daal::services::SyclExecutionContext(cl::sycl::queue(cl::sycl::default_selector()));
    }
    ~PySyclExecutionContext()
    {
        delete m_ctxt;
    }
private:
    daal::services::SyclExecutionContext *m_ctxt;
};


static std::string to_std_string(PyObject * o)
{
    return PyUnicode_AsUTF8(o);
}

#endif // __ONEAPI_H_INCLUDED__
