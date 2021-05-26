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

#define NO_IMPORT_ARRAY

#ifdef ONEDAL_DATA_PARALLEL
#include <CL/sycl.hpp>
#endif

#include "oneapi/dal/train.hpp"
#include "oneapi/dal/infer.hpp"
#include "oneapi/dal/compute.hpp"

#include "common/backend/utils.h"
#include "common/backend/policy_impl.h"

namespace oneapi::dal::python {

#define FuncCaller(func_name) [](auto&&... args) { return func_name(std::forward<decltype(args)>(args)...); }

class ONEDAL_BACKEND_EXPORT policy {
public:
    policy(const std::string& device_name);

    template <typename... Args>
    auto train(Args &&... args) const {
        return dispatch_and_run(FuncCaller(dal::train), std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto infer(Args &&... args) const {
        return dispatch_and_run(FuncCaller(dal::infer), std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto compute(Args &&... args) const {
        return dispatch_and_run(FuncCaller(dal::compute), std::forward<Args>(args)...);
    }

private:
    template <typename Func, typename... Args>
    auto dispatch_and_run(Func func, Args&&... args) const {

        if (impl_->get_kind() == host_policy_impl::kind()) {
            return func(std::forward<Args>(args)...);
        }
#ifdef ONEDAL_DATA_PARALLEL
        else if (impl_->get_kind() == data_parallel_policy_impl::kind()) {
            auto* impl_ptr = static_cast<data_parallel_policy_impl*>(impl_.get());
            sycl::queue queue = impl_ptr->get_queue();
            return func(queue, std::forward<Args>(args)...);
        }
#endif
        else {
            throw std::runtime_error("Unknown policy type");
        }
    }

    std::shared_ptr<policy_impl> impl_;
};

} // namespace oneapi::dal::python
