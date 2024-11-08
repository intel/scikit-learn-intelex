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
#include "onedal/common/policy_common.hpp"
#include "onedal/common/pybind11_helpers.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {

using host_policy_t = dal::detail::host_policy;
using default_host_policy_t = dal::detail::default_host_policy;

void instantiate_host_policy(py::module& m) {
    constexpr const char name[] = "host_policy";
    py::class_<host_policy_t> policy(m, name);
    policy.def(py::init<host_policy_t>());
    instantiate_host_policy(policy);
}

void instantiate_default_host_policy(py::module& m) {
    constexpr const char name[] = "default_host_policy";
    py::class_<default_host_policy_t> policy(m, name);
    policy.def(py::init<default_host_policy_t>());
    instantiate_host_policy(policy);
}

#ifdef ONEDAL_DATA_PARALLEL

using data_parallel_policy_t = dal::detail::data_parallel_policy;

void instantiate_data_parallel_policy(py::module& m) {
    constexpr const char name[] = "data_parallel_policy";
    py::class_<data_parallel_policy_t> policy(m, name);
    policy.def(py::init<data_parallel_policy_t>());
    policy.def(py::init([](std::uint32_t id) {
        return make_dp_policy(id);
    }));
    policy.def(py::init([](const std::string& filter) {
        return make_dp_policy(filter);
    }));
    policy.def(py::init([](const py::object& syclobj) {
        return make_dp_policy(syclobj);
    }));
    policy.def("get_device_id", [](const data_parallel_policy_t& policy) {
        return get_device_id(policy);
    });
    policy.def("get_device_name", [](const data_parallel_policy_t& policy) {
        return get_device_name(policy);
    });
    m.def("get_used_memory", &get_used_memory, py::return_value_policy::take_ownership);
}

class DummyQueue {
    public:
        DummyQueue(sycl::queue queue){
            _queue = queue;
            sycl_device = queue.get_device();
        }
        py::capsule _get_capsule(){
            return pack_queue(std::make_shared<sycl::queue>(_queue));
        }

        sycl::device sycl_device;

    private:
        sycl::queue _queue;
};

void instantiate_sycl_queue(py::module& m){
    py::class_<DummyQueue> syclqueue(m, "SyclQueue");
    syclqueue.def(py::init([](const py::object& syclobj) {
                return DummyQueue(get_queue_from_python(syclobj));
            })
        )
        .def(py::init([](const sycl::device& sycldevice) {
                return DummyQueue(sycl::queue{sycldevice});
            })
        )
        .def(py::init([](const std::string& filter) {
                return DummyQueue(get_queue_by_filter_string(filter));
            })
        )
        .def("_get_capsule", &DummyQueue::_get_capsule)
        .def_readonly("sycl_device", &DummyQueue::sycl_device);

    // expose limited sycl device features to python for oneDAL analysis
    py::class_<sycl::device> sycldevice(m, "SyclDevice");
        sycldevice.def(py::init([](std::uint32_t id) {
                return get_device_by_id(id).value();
            })
        )
        .def_property_readonly("has_aspect_fp64",[](const sycl::device& device) {
                return device.has(sycl::aspect::fp64);
            }
        )
        .def_property_readonly("has_aspect_fp16",[](const sycl::device& device) {
                return device.has(sycl::aspect::fp16);
            }
        )
        .def_property_readonly("filter_string",[](const sycl::device& device) {
                // assumes we are not working with accelerators
                std::string filter = device.is_cpu() ? "cpu:" : "gpu:";
                py::int_ id(get_device_id(device).value());
                return py::str(filter) + py::str(id);
            }
        )
        .def_property_readonly("is_cpu", &sycl::device::is_cpu)
        .def_property_readonly("is_gpu", &sycl::device::is_gpu);
}


#endif // ONEDAL_DATA_PARALLEL

ONEDAL_PY_INIT_MODULE(policy) {
    instantiate_host_policy(m);
    instantiate_default_host_policy(m);
    
#ifdef ONEDAL_DATA_PARALLEL
    instantiate_data_parallel_policy(m);
    instantiate_sycl_queue(m);
#endif // ONEDAL_DATA_PARALLEL
}

} // namespace oneapi::dal::python
