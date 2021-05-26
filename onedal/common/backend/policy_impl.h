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

#ifdef ONEDAL_DATA_PARALLEL
#include <CL/sycl.hpp>
#endif

namespace oneapi::dal::python {

class policy_impl {
public:
    virtual ~policy_impl() = default;
    virtual std::int64_t get_kind() const = 0;
};

class host_policy_impl : public policy_impl {
public:
    static std::int64_t kind();
    std::int64_t get_kind() const override { return kind(); }
};

#ifdef ONEDAL_DATA_PARALLEL

class data_parallel_policy_impl : public policy_impl {
public:
    static std::int64_t kind();

    data_parallel_policy_impl(const sycl::queue& queue);

    std::int64_t get_kind() const override { return kind(); }
    sycl::queue get_queue() const;

private:
    sycl::queue queue_;
};

#endif

} // namespace oneapi::dal::python
