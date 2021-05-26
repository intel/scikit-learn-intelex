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

#include "common/backend/policy.h"

namespace oneapi::dal::python {

static const std::int64_t host_policy_kind = 1;
static const std::int64_t data_parallel_policy_kind = 2;

std::int64_t host_policy_impl::kind() {
    return host_policy_kind;
}

#ifdef ONEDAL_DATA_PARALLEL
std::int64_t data_parallel_policy_impl::kind() {
    return data_parallel_policy_kind;
}

data_parallel_policy_impl::data_parallel_policy_impl(const sycl::queue& queue)
    : queue_(queue) {}

sycl::queue data_parallel_policy_impl::get_queue() const {
    return queue_;
}
#endif

} // namespace oneapi::dal::python
