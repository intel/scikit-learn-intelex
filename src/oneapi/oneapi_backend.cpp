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

#include "oneapi_backend.h"

PySyclExecutionContext::PySyclExecutionContext(const std::string & dev) : m_ctxt(NULL)
{
    if (dev == "gpu")
        m_ctxt = new daal::services::SyclExecutionContext(cl::sycl::queue(cl::sycl::gpu_selector()));
    else if (dev == "cpu")
        m_ctxt = new daal::services::SyclExecutionContext(cl::sycl::queue(cl::sycl::cpu_selector()));
    else if (dev == "host")
        m_ctxt = new daal::services::SyclExecutionContext(cl::sycl::queue(cl::sycl::host_selector()));
    else
    {
        throw std::runtime_error(std::string("Device is not supported: ") + dev);
    }
}

PySyclExecutionContext::~PySyclExecutionContext()
{
    daal::services::Environment::getInstance()->setDefaultExecutionContext(daal::services::CpuExecutionContext());
    delete m_ctxt;
    m_ctxt = NULL;
}

void PySyclExecutionContext::apply()
{
    daal::services::Environment::getInstance()->setDefaultExecutionContext(*m_ctxt);
}

#if INTEL_DAAL_VERSION >= 20210200
inline const sycl::queue& get_current_queue()
{
    auto& ctx = daal::services::Environment::getInstance()->getDefaultExecutionContext();
    auto* sycl_ctx = dynamic_cast<daal::services::internal::sycl::SyclExecutionContextImpl*>(&ctx);
    if (!sycl_ctx)
    {
        throw std::domain_error("Cannot get current queue outside sycl_context");
    }
    return sycl_ctx->getQueue();
}

// take a raw array and convert to usm pointer
template <typename T>
inline daal::services::SharedPtr<T>* to_usm(T * ptr, int * shape)
{
    auto queue = get_current_queue();
    const std::int64_t count = shape[0] * shape[1];
    T* usm_host_ptr = sycl::malloc_host<T>(count, queue);
    T* usm_device_ptr = sycl::malloc_device<T>(count, queue);
    if (!usm_host_ptr || !usm_device_ptr)
    {
        sycl::free(usm_host_ptr, queue);
        sycl::free(usm_device_ptr, queue);
        throw std::runtime_error("internal error during allocating USM memory");
    }

    // TODO: avoid using usm_host_ptr and copy directly to usm_device_ptr
    // It's a temporary solution till queue.memcpy() from non-usm memory does not work
    int res = daal::services::internal::daal_memcpy_s(usm_host_ptr, sizeof(T) * count, ptr, sizeof(T) * count);
    if (res)
    {
        sycl::free(usm_host_ptr, queue);
        sycl::free(usm_device_ptr, queue);
        throw std::runtime_error("internal error during data copying from host to USM memory");
    }

    try
    {
        auto event = queue.memcpy(usm_device_ptr, usm_host_ptr, sizeof(T) * count);
        event.wait_and_throw();
    }
    catch (std::exception& ex)
    {
        sycl::free(usm_host_ptr, queue);
        sycl::free(usm_device_ptr, queue);
        throw std::runtime_error("internal error during data copying from host to USM memory");
    }

    sycl::free(usm_host_ptr, queue);
    return new daal::services::SharedPtr<T>(usm_device_ptr, [q = queue](const void * data) {
            sycl::free(const_cast<void*>(data), q);
    });
}

template <typename T>
inline void del_usm(void * ptr)
{
    auto* sh_ptr = reinterpret_cast<daal::services::SharedPtr<T>*>(ptr);
    sh_ptr->reset();
    delete sh_ptr;
}
#endif

// take a raw array and convert to sycl buffer
template <typename T>
inline sycl::buffer<T>* to_sycl_buffer(T * ptr, int * shape)
{
    return new sycl::buffer<T>(ptr, sycl::range<1>(shape[0]*shape[1]), { sycl::property::buffer::use_host_ptr() });
}

template <typename T>
inline void del_sycl_buffer(void * ptr)
{
    auto* bf = reinterpret_cast<sycl::buffer<T>*>(ptr);
    delete bf;
}

template <typename T>
void* to_device(T * ptr, int * shape)
{
    #if INTEL_DAAL_VERSION >= 20210200
        return to_usm(ptr, shape);
    #else
        return to_sycl_buffer(ptr, shape);
    #endif
}

template <typename T>
void delete_device_data(void * ptr)
{
    #if INTEL_DAAL_VERSION >= 20210200
        del_usm<T>(ptr);
    #else
        del_sycl_buffer<T>(ptr);
    #endif
}

// take a sycl buffer and convert ti oneDAL NT
template <typename T, bool is_device_data>
daal::data_management::NumericTablePtr * to_daal_nt(void * ptr, int * shape)
{
    // ptr is SharedPtr<T>* in case of USM pointer
    // or just T* in case of host data
    // or sycl::buffer<T>* for previous oneDAL versions

    if constexpr(is_device_data)
    {
        typedef daal::data_management::SyclHomogenNumericTable<T> TBL_T;
#if INTEL_DAAL_VERSION >= 20210200
        auto* usm_ptr = reinterpret_cast<daal::services::SharedPtr<T>*>(ptr);
        // we need to return a pointer to safely cross language boundaries
        return new daal::data_management::NumericTablePtr(TBL_T::create(*usm_ptr, shape[1], shape[0], get_current_queue()));
#else
        auto* buffer = reinterpret_cast<sycl::buffer<T>*>(ptr);
        return new daal::data_management::NumericTablePtr(TBL_T::create(*buffer, shape[1], shape[0]));
#endif
    }
    else
    {
        typedef daal::data_management::HomogenNumericTable<T> TBL_T;
        auto* host_ptr = reinterpret_cast<T*>(ptr);
        // we need to return a pointer to safely cross language boundaries
        return new daal::data_management::NumericTablePtr(TBL_T::create(host_ptr, shape[1], shape[0]));
    }
}

// return a device data from a SyclHomogenNumericTable
template <typename T>
void * fromdaalnt(daal::data_management::NumericTablePtr * ptr)
{
    auto data = dynamic_cast<daal::data_management::SyclHomogenNumericTable<T> *>((*ptr).get());
    if (data)
    {
        daal::data_management::BlockDescriptor<T> block;
        data->getBlockOfRows(0, data->getNumberOfRows(), daal::data_management::readOnly, block);
        auto daalBuffer = block.getBuffer();

#if INTEL_DAAL_VERSION >= 20210200
        auto queue = get_current_queue();
        auto* usmPointer = new daal::services::SharedPtr<T>(daalBuffer.toUSM(queue, daal::data_management::readOnly));
        data->releaseBlockOfRows(block);
        return usmPointer;
#else
        auto* syclBuffer = new sycl::buffer<T>(daalBuffer.toSycl());
        data->releaseBlockOfRows(block);
        return syclBuffer;
#endif
    }
    return NULL;
}

template void* to_device(double * ptr, int * shape);
template void* to_device(float * ptr, int * shape);
template void* to_device(int * ptr, int * shape);

template void delete_device_data<double>(void * ptr);
template void delete_device_data<float>(void * ptr);
template void delete_device_data<int>(void * ptr);

template daal::data_management::NumericTablePtr * to_daal_nt<double, true>(void * ptr, int * shape);
template daal::data_management::NumericTablePtr * to_daal_nt<float, true>(void * ptr, int * shape);
template daal::data_management::NumericTablePtr * to_daal_nt<int, true>(void * ptr, int * shape);
template daal::data_management::NumericTablePtr * to_daal_nt<double, false>(void * ptr, int * shape);
template daal::data_management::NumericTablePtr * to_daal_nt<float, false>(void * ptr, int * shape);
template daal::data_management::NumericTablePtr * to_daal_nt<int, false>(void * ptr, int * shape);

template void * fromdaalnt<double>(daal::data_management::NumericTablePtr * ptr);
template void * fromdaalnt<float>(daal::data_management::NumericTablePtr * ptr);
template void * fromdaalnt<int>(daal::data_management::NumericTablePtr * ptr);
