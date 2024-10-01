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

#pragma once

#include <unordered_map>

namespace oneapi::dal::python {

template <typename Key, typename Value>
inline auto inverse_map(const std::unordered_map<Key, Value>& input)
    -> std::unordered_map<Value, Key> {
    const auto b_count = input.bucket_count();
    std::unordered_map<Value, Key> output(b_count);

    for (const auto& [key, value] : input) {
        output.emplace(value, key);
    }

    return output;
}

bool is_big_endian();
bool is_little_endian();

} // namespace oneapi::dal::python