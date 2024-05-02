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

#include <cstdint>
#include <optional>
#include <algorithm>

#include "oneapi/dal/detail/policy.hpp"

#include "onedal/common/device_lookup.hpp"

namespace oneapi::dal::python {

#ifdef ONEDAL_DATA_PARALLEL

const std::vector<sycl::device>& get_devices() {
    static const auto devices = sycl::device::get_devices();
    return devices;
}

template <typename Iter>
inline std::uint32_t get_id(Iter first, Iter it) {
    const auto raw_id = std::distance(first, it);
    return detail::integral_cast<std::uint32_t>(raw_id);
}

std::optional<std::uint32_t> get_device_id(const sycl::device& device) {
    const auto devices = get_devices();
    const auto first = devices.cbegin();
    const auto sentinel = devices.cend();
    auto iter = std::find(first, sentinel, device);
    if (iter != sentinel) {
        return get_id(first, iter);
    }
    else {
        return {};
    }
}

std::optional<sycl::device> get_device_by_id(std::uint32_t device_id) {
    auto casted = detail::integral_cast<std::size_t>(device_id);
    const auto devices = get_devices();
    if (casted < devices.size()) {
        return devices.at(casted);
    }
    else {
        return {};
    }
}

#endif // ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::python
