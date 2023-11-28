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

#include <pybind11/pybind11.h>

#include "oneapi/dal/train.hpp"
#include "oneapi/dal/infer.hpp"
#include "oneapi/dal/compute.hpp"

#define ONEDAL_PARAM_DISPATCH_VALUE(value, value_case, ops, ...) \
    if (value == value_case) {                                   \
        return ops.template operator()<__VA_ARGS__>(params);     \
    }                                                            \
    else

#define ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(name) \
    { throw std::runtime_error("Invalid value for parameter <" #name ">"); }


namespace oneapi::dal::python {

template <typename Ops>
struct fptype2t {
    fptype2t(const Ops& ops) : ops(ops) {}

    auto operator()(const pybind11::dict& params) {
        const auto fptype = params["fptype"].cast<std::string>();
        ONEDAL_PARAM_DISPATCH_VALUE(fptype, "float", ops, float);
        ONEDAL_PARAM_DISPATCH_VALUE(fptype, "double", ops, double);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(fptype);
    }

    Ops ops;
};

template <typename Policy, typename Input, typename Ops>
struct compute_ops {
    using Task = typename Input::task_t;

    compute_ops(const Policy& policy, const Input& input, const Ops& ops)
        : policy(policy),
          input(input),
          ops(ops) {}

    template <typename Float, typename Method, typename... Args>
    auto operator()(const pybind11::dict& params) {
        auto desc = ops.template operator()<Float, Method, Task, Args...>(params);
        return dal::compute(policy, desc, input);
    }

    Policy policy;
    Input input;
    Ops ops;
};

template <typename Policy, typename Input, typename Ops>
struct train_ops {
    using Task = typename Input::task_t;

    train_ops(const Policy& policy, const Input& input, const Ops& ops)
        : policy(policy),
          input(input),
          ops(ops) {}

    template <typename Float, typename Method, typename... Args>
    auto operator()(const pybind11::dict& params) {
        auto desc = ops.template operator()<Float, Method, Task, Args...>(params);
        return dal::train(policy, desc, input);
    }

    Policy policy;
    Input input;
    Ops ops;
};

template <typename Policy, typename Input, typename Ops, typename Hyperparams>
struct train_ops_with_hyperparams {
    using Task = typename Input::task_t;

    train_ops_with_hyperparams(
        const Policy& policy, const Input& input,
        const Ops& ops, const Hyperparams& hyperparams)
        : policy(policy),
          input(input),
          ops(ops),
          hyperparams(hyperparams) {}

    template <typename Float, typename Method, typename... Args>
    auto operator()(const pybind11::dict& params) {
        auto desc = ops.template operator()<Float, Method, Task, Args...>(params);
        return dal::train(policy, desc, hyperparams, input);
    }

    Policy policy;
    Input input;
    Ops ops;
    Hyperparams hyperparams;
};

template <typename Policy, typename Input, typename Ops>
struct infer_ops {
    using Task = typename Input::task_t;

    infer_ops(const Policy& policy, const Input& input, const Ops& ops)
        : policy(policy),
          input(input),
          ops(ops) {}

    template <typename Float, typename Method, typename... Args>
    auto operator()(const pybind11::dict& params) {
        auto desc = ops.template operator()<Float, Method, Task, Args...>(params);
        return dal::infer(policy, desc, input);
    }

    Policy policy;
    Input input;
    Ops ops;
};

} // namespace oneapi::dal::python
