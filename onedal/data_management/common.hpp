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

#pragma once

#include <string>
#include <stdexcept>

namespace oneapi::dal::python {

template <typename Type>
struct range {
    range(Type l, Type r) noexcept(false) : left{ l }, right{ r } {
        this->check_correctness();
    }

    void check_correctness() const noexcept(false) {
        const std::string msg = "Boundaries are unordered";
        if (right < left)
            throw std::range_error(msg);
    }

    const Type left, right;
};

template <typename Type, typename Range = range<Type>>
inline void check_in_range(const Range& inner, const Range& outer) {
    const std::string msg = "Inner & outer ranges are bad";
    const auto& [l_i, r_i] = inner;
    const auto& [l_o, r_o] = outer;
    inner.check_correctness();
    outer.check_correctness();
    if (l_i < l_o || r_o < r_i)
        throw std::range_error(msg);
}

} // namespace oneapi::dal::python