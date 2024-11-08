/*******************************************************************************
* Copyright 2024 Intel Corporation
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
#include "onedal/common/sycl_interfaces.hpp"
#include "onedal/common/pybind11_helpers.hpp"

#ifdef ONEDAL_DATA_PARALLEL

class SyclQueue {
    public:
        SyclQueue(sycl::queue queue){
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

void instantiate_sycl_interfaces(py::module& m){
    py::class_<SyclQueue> syclqueue(m, "SyclQueue");
    syclqueue.def(py::init([](const sycl::device& sycldevice) {
                return SyclQueue(sycl::queue{sycldevice});
            })
        )
        .def(py::init([](const std::string& filter) {
                return SyclQueue(get_queue_by_filter_string(filter));
            })
        )
        
        .def("_get_capsule", &SyclQueue::_get_capsule)
        .def_readonly("sycl_device", &SyclQueue::sycl_device);

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
                std::string filter = get_device_name(device);
                py::int_ id(get_device_id(device).value());
                return py::str(filter) + py::str(id);
            }
        )
        .def_property_readonly("is_cpu", &sycl::device::is_cpu)
        .def_property_readonly("is_gpu", &sycl::device::is_gpu);
}



ONEDAL_PY_INIT_MODULE(sycl) {
    instantiate_sycl_interfaces(m);
}
#endif
