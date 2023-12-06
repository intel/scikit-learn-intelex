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

#include "oneapi/dal/detail/policy.hpp"
#include "onedal/common/pybind11_helpers.hpp"
#ifdef ONEDAL_DATA_PARALLEL
#include "services/internal/sycl/math/mkl_dal_sycl.hpp"
#endif

namespace py = pybind11;

namespace oneapi::dal::python {

ONEDAL_PY_INIT_MODULE(policy) {
    py::class_<detail::host_policy>(m, "host_policy")
        .def(py::init())
        .def("get_device_name", [](const detail::host_policy& self) {
            return "cpu";
        });

#ifdef ONEDAL_DATA_PARALLEL
    py::class_<detail::data_parallel_policy>(m, "data_parallel_policy")
        .def(py::init([](std::size_t address_of_queue) {
            auto* queue = reinterpret_cast<sycl::queue*>(address_of_queue);
            return detail::data_parallel_policy(*queue);
        }))
        .def(py::init([](const std::string& filter_string) {
            sycl::queue q { sycl::ext::oneapi::filter_selector(filter_string) };
            return detail::data_parallel_policy(q);
        }))
        .def("get_device_name", [](const detail::data_parallel_policy& self) {
            if (self.get_queue().get_device().is_gpu()) {
                return "gpu";
            } else if (self.get_queue().get_device().is_cpu()) {
                return "cpu";
            }
            return "unknown";
        });

    // mkl blas compute mode will be linked to oneDAL via data parallel policy in the future
    py::enum_<oneapi::fpk::blas::compute_mode>(m, "ComputeMode")
        .value("unset",            oneapi::fpk::blas::compute_mode::unset)
        .value("float_to_bf16",    oneapi::fpk::blas::compute_mode::float_to_bf16)
        .value("float_to_bf16x2",  oneapi::fpk::blas::compute_mode::float_to_bf16x2)
        .value("float_to_bf16x3",  oneapi::fpk::blas::compute_mode::float_to_bf16x3)
        .value("float_to_tf32",    oneapi::fpk::blas::compute_mode::float_to_tf32)
        .value("complex_3m",       oneapi::fpk::blas::compute_mode::complex_3m)
        .value("any",              oneapi::fpk::blas::compute_mode::any)
        .value("standard",         oneapi::fpk::blas::compute_mode::standard)
        .value("prefer_alternate", oneapi::fpk::blas::compute_mode::prefer_alternate)
        .value("force_alternate",  oneapi::fpk::blas::compute_mode::force_alternate)
        .export_values();
#endif
}

} // namespace oneapi::dal::python
