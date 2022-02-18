/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "dpctl_interop/daal_context_service.h"

#include "daal_sycl.h"
#include "dppl_sycl_types.h"
#include "dppl_sycl_queue_manager.h"

#ifndef DAAL_SYCL_INTERFACE
#include <type_traits>
static_assert(false, "DAAL_SYCL_INTERFACE not defined")
#endif

void _dppl_set_current_queue_to_daal_context()
{
    auto dppl_queue = DPPLQueueMgr_GetCurrentQueue();
    if(dppl_queue != NULL)
    {
        cl::sycl::queue * sycl_queue = reinterpret_cast<cl::sycl::queue*>(dppl_queue);

        daal::services::SyclExecutionContext ctx (*sycl_queue);
        daal::services::Environment::getInstance()->setDefaultExecutionContext(ctx);
    }
    else
    {
        throw std::runtime_error("Cannot set daal context: Pointer to queue object is NULL");
    }
}

void _dppl_reset_daal_context()
{
    daal::services::CpuExecutionContext ctx;
    daal::services::Environment::getInstance()->setDefaultExecutionContext(ctx);
}
