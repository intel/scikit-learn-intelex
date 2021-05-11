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

#include "oneapi/dal/detail/serialization.hpp"
#include "Python.h"

class mock_archive_entry {
public:
    explicit mock_archive_entry(const std::uint8_t* data, data_type dtype) : dtype_(dtype) {
        const std::int64_t dtype_size = detail::get_data_type_size(dtype);
        data_.assign(data, data + dtype_size);
    }

    template <typename T>
    T get() const {
        using trivial_t = detail::trivial_serialization_type_t<T>;
        if (dtype_ != detail::make_data_type<trivial_t>()) {
            throw std::invalid_argument{ "Data types do not match" };
        }
        const T* data = reinterpret_cast<const T*>(data_.data());
        return *data;
    }

    void write_to(std::uint8_t* dst, data_type dtype) const {
        if (dtype != dtype_) {
            throw std::invalid_argument{ "Data types do not match" };
        }
        std::copy(data_.begin(), data_.end(), dst);
    }

private:
    std::vector<std::uint8_t> data_;
    data_type dtype_;
};

class mock_archive_state {
public:
    using entries_t = std::vector<mock_archive_entry>;

    mock_archive_state() : entries_(new entries_t{}) {}

    const entries_t& get() const {
        return *entries_;
    }

    entries_t& get() {
        return *entries_;
    }

    template <typename T>
    T get(std::int64_t index) const {
        ONEDAL_ASSERT(index < get_count());
        return get()[index].template get<T>();
    }

    std::int64_t get_count() const {
        return std::int64_t(get().size());
    }

private:
    std::shared_ptr<entries_t> entries_;
};

class mock_input_archive {
public:
    std::int64_t prologue_call_count = 0;
    std::int64_t epilogue_call_count = 0;

    explicit mock_input_archive(const mock_archive_state& state) : state_(state) {}

    void operator()(void* data, data_type dtype, std::int64_t count = 1) {
        check_possition();

        for (std::int64_t i = 0; i < count; i++) {
            load(i, data, dtype);
        }
    }

    void prologue() {
        prologue_call_count++;
    }

    void epilogue() {
        epilogue_call_count++;
    }

private:
    void check_possition() const {
        if (position_ >= state_.get().size()) {
            throw std::runtime_error{ "We reached the end of input stream" };
        }
    }

    void load(std::int64_t index, void* data, data_type dtype) {
        const std::int64_t dtype_size = detail::get_data_type_size(dtype);
        std::uint8_t* data_bytes = static_cast<std::uint8_t*>(data);
        state_.get()[position_++].write_to(data_bytes + index * dtype_size, dtype);
    }

    mock_archive_state state_;
    std::size_t position_ = 0;
};

class mock_output_archive {
public:
    std::int64_t prologue_call_count = 0;
    std::int64_t epilogue_call_count = 0;

    explicit mock_output_archive(const mock_archive_state& state) : state_(state) {}

    void operator()(const void* data, data_type dtype, std::int64_t count = 1) {
        state_.get().reserve(state_.get().size() + count);

        for (std::int64_t i = 0; i < count; i++) {
            save(i, data, dtype);
        }
    }

    void prologue() {
        prologue_call_count++;
    }

    void epilogue() {
        epilogue_call_count++;
    }

private:
    void save(std::int64_t index, const void* data, data_type dtype) {
        const std::int64_t dtype_size = detail::get_data_type_size(dtype);
        const std::uint8_t* data_bytes = static_cast<const std::uint8_t*>(data);
        state_.get().emplace_back(data_bytes + index * dtype_size, dtype);
    }

    mock_archive_state state_;
};

template <typename T>
PyObject* serialize_si(const T& original) {
    binary_input_archive ar;
    detail::serialize(original, ar);

    daal::data_management::InputDataArchive dataArch;

    output_archive internal_archive{ archive };
    detail::serialize(original, ar);

    (daal::services::staticPointerCast<daal::data_management::SerializationIface>(*ptr))
        ->serialize(dataArch);

    Py_ssize_t buf_len = dataArch.getSizeOfArchive();
    daal::services::SharedPtr<daal::byte> shr_ptr = dataArch.getArchiveAsArraySharedPtr();

    return PyBytes_FromStringAndSize(reinterpret_cast<std::uint8_t*>(shr_ptr.get()), buf_len);
}

template <typename T>
T deserialize_si(PyObject* py_bytes) {
    T deserialized;
    std::uint8_t* buf;
    Py_ssize_t buf_len;
    PyBytes_AsStringAndSize(py_bytes, &buf, &buf_len);
    mock_archive_state state;

    mock_input_archive ar(state);
    detail::deserialize(deserialized, ar);
    return deserialized;

    // daal::data_management::OutputDataArchive dataArch(reinterpret_cast<std::uint8_t *>(buf), buf_len);

    // return new T(
    //     daal::services::staticPointerCast<typename T::ElementType>(dataArch.getAsSharedPtr()));
}
