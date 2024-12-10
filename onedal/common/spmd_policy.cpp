/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifdef ONEDAL_DATA_PARALLEL_SPMD

#include "oneapi/dal/detail/spmd_policy.hpp"
#include "oneapi/dal/spmd/mpi/communicator.hpp"

#include "onedal/common/policy.hpp"
#include "onedal/common/pybind11_helpers.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {

using dp_policy_t = dal::detail::data_parallel_policy;
using spmd_policy_t = dal::detail::spmd_policy<dp_policy_t>;

inline spmd_policy_t make_spmd_policy(dp_policy_t&& local) {
    sycl::queue& queue = local.get_queue();
    using backend_t = dal::preview::spmd::backend::mpi;
    auto comm = dal::preview::spmd::make_communicator<backend_t>(queue);
    return spmd_policy_t{ std::forward<dp_policy_t>(local), std::move(comm) };
}

template <typename... Args>
inline spmd_policy_t make_spmd_policy(Args&&... args) {
    auto local = make_dp_policy(std::forward<Args>(args)...);
    return make_spmd_policy(std::move(local));
}

template <typename Arg, typename Policy = spmd_policy_t>
inline void instantiate_costructor(py::class_<Policy>& policy) {
    policy.def(py::init([](const Arg& arg) {
        return make_spmd_policy(arg);
    }));
}

void instantiate_spmd_policy(py::module& m) {
    constexpr const char name[] = "spmd_data_parallel_policy";
    py::class_<spmd_policy_t> policy(m, name);
    policy.def(py::init<spmd_policy_t>());
    policy.def(py::init([](const dp_policy_t& local) {
        return make_spmd_policy(local);
    }));
    policy.def(py::init([](std::uint32_t id) {
        return make_spmd_policy(id);
    }));
    policy.def(py::init([](const std::string& filter) {
        return make_spmd_policy(filter);
    }));
    policy.def(py::init([](const py::object& syclobj) {
        return make_spmd_policy(syclobj);
    }));
    policy.def("get_device_id", [](const spmd_policy_t& policy) {
        return get_device_id(policy.get_local());
    });
    policy.def("get_device_name", [](const spmd_policy_t& policy) {
        return get_device_name(policy.get_local());
    });
}

ONEDAL_PY_INIT_MODULE(spmd_policy) {
    instantiate_spmd_policy(m);
} // ONEDAL_PY_INIT_MODULE(spmd_policy)

} // namespace oneapi::dal::python

#endif // ONEDAL_DATA_PARALLEL_SPMD
