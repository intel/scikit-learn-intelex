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
******************************************************************************/

#ifndef __ONEAPI_H_INCLUDED__
#define __ONEAPI_H_INCLUDED__

#include "daal_sycl.h"
#ifndef DAAL_SYCL_INTERFACE
#include <type_traits>
#include <memory>
static_assert(false, "DAAL_SYCL_INTERFACE not defined")
#endif

#include "numpy/ndarraytypes.h"
#include "oneapi_api.h"

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
        else if(dev == "host") m_ctxt = new daal::services::SyclExecutionContext(cl::sycl::queue(cl::sycl::host_selector()));
        else
        {
            throw std::runtime_error(std::string("Device is not supported: ") + dev);
        }
        daal::services::Environment::getInstance()->setDefaultExecutionContext(*m_ctxt);
    }
    ~PySyclExecutionContext()
    {
        daal::services::Environment::getInstance()->setDefaultExecutionContext(daal::services::CpuExecutionContext());
        delete m_ctxt;
    }
private:
    daal::services::SyclExecutionContext *m_ctxt;
};


static std::string to_std_string(PyObject * o)
{
    return PyUnicode_AsUTF8(o);
}

void set_daal_context(PyObject* o)
{
    if(PyCapsule_IsValid(o, NULL) == 0) { throw std::runtime_error("Cannot set daal context: invalid queue object"); }
    void * ptr = PyCapsule_GetPointer(o, NULL);
    if(PyErr_Occurred()) { PyErr_Print(); throw std::runtime_error("Python Error"); }

    auto* q = (std::shared_ptr<cl::sycl::queue>*)ptr;
    daal::services::SyclExecutionContext ctx (*(*q));
    daal::services::Environment::getInstance()->setDefaultExecutionContext(ctx);
}

void release_sycl_queue(PyObject* o)
{
    if(PyCapsule_IsValid(o, NULL) == 0) { throw std::runtime_error("Cannot release daal context: invalid queue object"); }
    void * ptr = PyCapsule_GetPointer(o, NULL);

    if(PyErr_Occurred()) { PyErr_Print(); throw std::runtime_error("Python Error"); }
    delete ptr;

    Py_DECREF(o);
}


// take a raw array and convert to sycl buffer
template<typename T>
inline cl::sycl::buffer<T, 1> * tosycl(T * ptr, int * shape)
{
    daal::services::Buffer<T> buff(ptr, shape[0]*shape[1]);
    // we need to return a pointer to safely cross language boundaries
    return new cl::sycl::buffer<T, 1>(buff.toSycl());
}

static void * tosycl(void * ptr, int typ, int * shape)
{
    switch(typ) {
    case NPY_DOUBLE:
        return tosycl(reinterpret_cast<double*>(ptr), shape);
        break;
    case NPY_FLOAT:
        return tosycl(reinterpret_cast<float*>(ptr), shape);
        break;
    case NPY_INT:
        return tosycl(reinterpret_cast<int*>(ptr), shape);
        break;
    default: throw std::invalid_argument("invalid input array type (must be double, float or int)");
    }
}

static void del_scl_buffer(void * ptr, int typ)
{
    if(!ptr) return;
    switch(typ) {
    case NPY_DOUBLE:
        delete reinterpret_cast<cl::sycl::buffer<double, 1>*>(ptr);
        break;
    case NPY_FLOAT:
        delete reinterpret_cast<cl::sycl::buffer<float, 1>*>(ptr);
        break;
    case NPY_INT:
        delete reinterpret_cast<cl::sycl::buffer<int, 1>*>(ptr);
        break;
    default: throw std::invalid_argument("invalid input array type (must be double, float or int)");
    }
}

// take a sycl buffer and convert ti DAAL NT
template<typename T>
inline daal::services::SharedPtr<daal::data_management::SyclHomogenNumericTable<T> > * todaalnt(T * ptr, int * shape)
{
    typedef daal::data_management::SyclHomogenNumericTable<T> TBL_T;
    // we need to return a pointer to safely cross language boundaries
    return new daal::services::SharedPtr<TBL_T>(TBL_T::create(*reinterpret_cast<cl::sycl::buffer<T, 1>*>(ptr), shape[1], shape[0]));
}

static void * todaalnt(void* ptr, int typ, int * shape)
{
    switch(typ) {
    case NPY_DOUBLE:
        return todaalnt(reinterpret_cast<double*>(ptr), shape);
        break;
    case NPY_FLOAT:
        return todaalnt(reinterpret_cast<float*>(ptr), shape);
        break;
    case NPY_INT:
        return todaalnt(reinterpret_cast<int*>(ptr), shape);
        break;
    default: throw std::invalid_argument("invalid input array type (must be double, float or int)");
    }
}

// return a sycl::buffer from a SyclHomogenNumericTable
template<typename T>
inline cl::sycl::buffer<T, 1> * fromdaalnt(daal::data_management::NumericTablePtr * ptr)
{
    auto data = dynamic_cast< daal::data_management::SyclHomogenNumericTable< T >* >((*ptr).get());
    if(data) {
        daal::data_management::BlockDescriptor<T> block;
        data->getBlockOfRows(0, data->getNumberOfRows(), daal::data_management::readOnly, block); //data is NumericTable object
        auto daalBuffer = block.getBuffer();
        auto syclBuffer = new cl::sycl::buffer<T, 1>(daalBuffer.toSycl());
        data->releaseBlockOfRows(block);
        return syclBuffer;
    }
    return NULL;
}

void * c_make_py_from_sycltable(void * _ptr, int typ)
{
    auto ptr = reinterpret_cast<daal::data_management::NumericTablePtr *>(_ptr);

    switch(typ) {
    case NPY_DOUBLE:
        return fromdaalnt<double>(ptr);
        break;
    case NPY_FLOAT:
        return fromdaalnt<float>(ptr);
        break;
    case NPY_INT:
        return fromdaalnt<int>(ptr);
        break;
    default: throw std::invalid_argument("invalid output array type (must be double, float or int)");
    }
    return NULL;
}

#endif // __ONEAPI_H_INCLUDED__
