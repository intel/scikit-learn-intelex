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

#include <stdexcept>
#include <string>

#include <Python.h>

#include "backend/data/data.h"

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/detail/homogen_utils.hpp"
#include "oneapi/dal/table/detail/csr.hpp"

#ifdef ONEDAL_DATA_PARALLEL
#include <CL/sycl.hpp>
#endif

#ifdef DPCTL_ENABLE
#include "dpctl_sycl_types.h"
#include "dpctl_sycl_queue_manager.h"
#endif

namespace oneapi::dal::python {
#define is_array(a)         ((a) && PyArray_Check(a))
#define array_type(a)       PyArray_TYPE((PyArrayObject *)a)
#define array_is_behaved(a) (PyArray_ISCARRAY_RO((PyArrayObject *)a) && array_type(a) < NPY_OBJECT)
#define array_is_behaved_F(a) \
    (PyArray_ISFARRAY_RO((PyArrayObject *)a) && array_type(a) < NPY_OBJECT)
#define array_is_native(a) (PyArray_ISNOTSWAPPED((PyArrayObject *)a))
#define array_numdims(a)   PyArray_NDIM((PyArrayObject *)a)
#define array_data(a)      PyArray_DATA((PyArrayObject *)a)
#define array_size(a, i)   PyArray_DIM((PyArrayObject *)a, i)

#ifdef ONEDAL_DATA_PARALLEL
int init_numpy() {
    import_array();
    return 0;
}

const static int numpy_initialized = init_numpy();
#endif

template <typename T>
inline dal::homogen_table convert_to_homogen_impl(PyArrayObject *np_data) {
    std::int64_t column_count = 1;

    if (array_numdims(np_data) > 2) {
        throw std::runtime_error("Input array has wrong dimensionality (must be 2d).");
    }
    T *data_pointer = reinterpret_cast<T *>(array_data(np_data));
    // TODO: check safe cast from int to std::int64_t
    const std::int64_t row_count = static_cast<std::int64_t>(array_size(np_data, 0));
    if (array_numdims(np_data) == 2) {
        // TODO: check safe cast from int to std::int64_t
        column_count = static_cast<std::int64_t>(array_size(np_data, 1));
    }
    const auto layout =
        array_is_behaved_F(np_data) ? dal::data_layout::column_major : dal::data_layout::row_major;
    auto res_table = dal::homogen_table(data_pointer,
                                        row_count,
                                        column_count,
                                        dal::detail::empty_delete<const T>(),
                                        layout);
    // we need it increment the ref-count if we use the input array in-place
    // if we copied/converted it we already own our own reference
    if (reinterpret_cast<PyArrayObject *>(data_pointer) == np_data)
        Py_INCREF(np_data);
    return res_table;
}

template <typename T>
inline dal::detail::csr_table convert_to_csr_impl(PyObject *py_data,
                                                  PyObject *py_column_indices,
                                                  PyObject *py_row_indices,
                                                  std::int64_t row_count,
                                                  std::int64_t column_count) {
    PyArrayObject *np_data = reinterpret_cast<PyArrayObject *>(py_data);
    PyArrayObject *np_column_indices = reinterpret_cast<PyArrayObject *>(py_column_indices);
    PyArrayObject *np_row_indices = reinterpret_cast<PyArrayObject *>(py_row_indices);

    const std::int64_t *row_indices_zero_based =
        static_cast<std::int64_t *>(array_data(np_row_indices));
    const std::int64_t row_indices_count = static_cast<std::int64_t>(array_size(np_row_indices, 0));

    auto row_indices_one_based = dal::array<std::int64_t>::empty(row_indices_count);
    auto row_indices_one_based_data = row_indices_one_based.get_mutable_data();

    for (std::int64_t i = 0; i < row_indices_count; ++i)
        row_indices_one_based_data[i] = row_indices_zero_based[i] + 1;

    const std::int64_t *column_indices_zero_based =
        static_cast<std::int64_t *>(array_data(np_column_indices));
    const std::int64_t column_indices_count =
        static_cast<std::int64_t>(array_size(np_column_indices, 0));

    auto column_indices_one_based = dal::array<std::int64_t>::empty(column_indices_count);
    auto column_indices_one_based_data = column_indices_one_based.get_mutable_data();

    for (std::int64_t i = 0; i < column_indices_count; ++i)
        column_indices_one_based_data[i] = column_indices_zero_based[i] + 1;

    const T *data_pointer = static_cast<T *>(array_data(np_data));
    const std::int64_t data_count = static_cast<std::int64_t>(array_size(np_data, 0));

    auto res_table = dal::detail::csr_table(
        dal::array<T>(data_pointer, data_count, dal::detail::empty_delete<const T>()),
        column_indices_one_based,
        row_indices_one_based,
        row_count,
        column_count);
    return res_table;
}

dal::table convert_to_table(PyObject *obj) {
    dal::table res;
    if (obj == nullptr || obj == Py_None) {
        return res;
    }
    if (is_array(obj)) {
        PyArrayObject *ary = reinterpret_cast<PyArrayObject *>(obj);
        if (array_is_behaved(ary) || array_is_behaved_F(ary)) {
#define MAKE_HOMOGEN_TABLE(CType) res = convert_to_homogen_impl<CType>(ary);
            SET_NPY_FEATURE(PyArray_DESCR(ary)->type,
                            MAKE_HOMOGEN_TABLE,
                            throw std::invalid_argument("Found unsupported array type"));
#undef MAKE_HOMOGEN_TABLE
        }
        else {
            throw std::invalid_argument(
                "[convert_to_table] Numpy input Could not convert Python object to onedal table.");
        }
    }
    else if (strcmp(Py_TYPE(obj)->tp_name, "csr_matrix") == 0) {
        PyObject *py_data = PyObject_GetAttrString(obj, "data");
        PyObject *py_column_indices = PyObject_GetAttrString(obj, "indices");
        PyObject *py_row_indices = PyObject_GetAttrString(obj, "indptr");

        PyObject *py_shape = PyObject_GetAttrString(obj, "shape");
        if (!(is_array(py_data) && is_array(py_column_indices) && is_array(py_row_indices) &&
              array_numdims(py_data) == 1 && array_numdims(py_column_indices) == 1 &&
              array_numdims(py_row_indices) == 1)) {
            throw std::invalid_argument("[convert_to_table] Got invalid csr_matrix object.");
        }
        PyObject *np_data = PyArray_FROMANY(py_data, array_type(py_data), 0, 0, NPY_ARRAY_CARRAY);
        PyObject *np_column_indices =
            PyArray_FROMANY(py_column_indices,
                            NPY_UINT64,
                            0,
                            0,
                            NPY_ARRAY_CARRAY | NPY_ARRAY_ENSURECOPY | NPY_ARRAY_FORCECAST);
        PyObject *np_row_indices =
            PyArray_FROMANY(py_row_indices,
                            NPY_UINT64,
                            0,
                            0,
                            NPY_ARRAY_CARRAY | NPY_ARRAY_ENSURECOPY | NPY_ARRAY_FORCECAST);

        PyObject *np_row_count = PyTuple_GetItem(py_shape, 0);
        PyObject *np_column_count = PyTuple_GetItem(py_shape, 1);
        if (!(np_data && np_column_indices && np_row_indices && np_row_count && np_column_count)) {
            throw std::invalid_argument(
                "[convert_to_table] Failed accessing csr data when converting csr_matrix.\n");
        }

        const std::int64_t row_count = static_cast<std::int64_t>(PyLong_AsSsize_t(np_row_count));
        const std::int64_t column_count =
            static_cast<std::int64_t>(PyLong_AsSsize_t(np_column_count));

#define MAKE_CSR_TABLE(CType)                           \
    res = convert_to_csr_impl<CType>(np_data,           \
                                     np_column_indices, \
                                     np_row_indices,    \
                                     row_count,         \
                                     column_count);
        SET_NPY_FEATURE(array_type(np_data),
                        MAKE_CSR_TABLE,
                        throw std::invalid_argument("Found unsupported data type in csr_matrix"));
#undef MAKE_CSR_TABLE
    }
    else {
        throw std::invalid_argument(
            "[convert_to_table] Not available input format for convert Python object to onedal table.");
    }
    return res;
}

void free_capsule(PyObject *cap) {
    // TODO: check safe cast
    dal::base *stored_array = static_cast<dal::base *>(PyCapsule_GetPointer(cap, NULL));
    if (stored_array) {
        delete stored_array;
    }
}

template <int NpType, typename T = byte_t>
static PyObject *convert_to_numpy_impl(dal::array<T> &array,
                                       std::int64_t row_count,
                                       std::int64_t column_count = 0) {
    const std::int64_t size_dims = column_count == 0 ? 1 : 2;

    npy_intp dims[2] = { static_cast<npy_intp>(row_count), static_cast<npy_intp>(column_count) };
    array.need_mutable_data();
    PyObject *obj = PyArray_SimpleNewFromData(size_dims,
                                              dims,
                                              NpType,
                                              static_cast<void *>(array.get_mutable_data()));
    if (!obj)
        throw std::invalid_argument("Conversion to numpy array failed");

    void *opaque_value = static_cast<void *>(new dal::array<T>(array));
    PyObject *cap = PyCapsule_New(opaque_value, NULL, free_capsule);
    PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(obj), cap);
    return obj;
}

template <int NpType, typename T>
static PyObject *convert_to_py_from_csr_impl(const detail::csr_table &table) {
    PyObject *result = PyTuple_New(3);
    const std::int64_t rows_indices_count = table.get_row_count() + 1;

    const std::int64_t *row_indices_one_based = table.get_row_indices();
    std::uint64_t *row_indices_zero_based_data =
        detail::host_allocator<std::uint64_t>().allocate(rows_indices_count);
    for (std::int64_t i = 0; i < rows_indices_count; ++i)
        row_indices_zero_based_data[i] = row_indices_one_based[i] - 1;

    auto row_indices_zero_based_array =
        dal::array<std::uint64_t>::wrap(row_indices_zero_based_data, rows_indices_count);
    PyObject *py_row =
        convert_to_numpy_impl<NPY_UINT64, std::uint64_t>(row_indices_zero_based_array,
                                                         rows_indices_count);
    PyTuple_SetItem(result, 2, py_row);

    const std::int64_t non_zero_count = row_indices_zero_based_data[rows_indices_count - 1];
    const T *data = reinterpret_cast<const T *>(table.get_data());
    auto data_array = dal::array<T>::wrap(data, non_zero_count);

    PyObject *py_data = convert_to_numpy_impl<NpType, T>(data_array, non_zero_count);
    PyTuple_SetItem(result, 0, py_data);

    const std::int64_t *column_indices_one_based = table.get_column_indices();
    std::uint64_t *column_indices_zero_based_data =
        detail::host_allocator<std::uint64_t>().allocate(non_zero_count);
    for (std::int64_t i = 0; i < non_zero_count; ++i)
        column_indices_zero_based_data[i] = column_indices_one_based[i] - 1;

    auto column_indices_zero_based_array =
        dal::array<std::uint64_t>::wrap(column_indices_zero_based_data, non_zero_count);
    PyObject *py_col =
        convert_to_numpy_impl<NPY_UINT64, std::uint64_t>(column_indices_zero_based_array,
                                                         non_zero_count);
    PyTuple_SetItem(result, 1, py_col);
    return result;
}

PyObject *convert_to_numpy(const dal::table &input) {
    PyObject *res = nullptr;
    if (!input.has_data()) {
        throw std::invalid_argument("Empty data to output result from oneDAL");
    }
    if (input.get_kind() == dal::homogen_table::kind()) {
        const auto &homogen_input = static_cast<const dal::homogen_table &>(input);
        if (homogen_input.get_data_layout() == dal::data_layout::row_major) {
            const dal::data_type dtype = homogen_input.get_metadata().get_data_type(0);

#define MAKE_NYMPY_FROM_HOMOGEN(NpType)                                        \
    {                                                                          \
        auto bytes_array = dal::detail::get_original_data(homogen_input);      \
        res = convert_to_numpy_impl<NpType>(bytes_array,                       \
                                            homogen_input.get_row_count(),     \
                                            homogen_input.get_column_count()); \
    }
            SET_CTYPE_NPY_FROM_DAL_TYPE(
                dtype,
                MAKE_NYMPY_FROM_HOMOGEN,
                throw std::invalid_argument("Not avalible to convert a numpy"));
#undef MAKE_NYMPY_FROM_HOMOGEN
        }
        else {
            throw std::invalid_argument(
                "Output oneDAL table doesn't have row major format for homogen table");
        }
    }
    else if (input.get_kind() == dal::detail::csr_table::kind()) {
        const auto &csr_input = static_cast<const detail::csr_table &>(input);
        const dal::data_type dtype = csr_input.get_metadata().get_data_type(0);
#define MAKE_PY_FROM_CSR(NpType, T) \
    { res = convert_to_py_from_csr_impl<NpType, T>(csr_input); }
        SET_CTYPES_NPY_FROM_DAL_TYPE(
            dtype,
            MAKE_PY_FROM_CSR,
            throw std::invalid_argument("Not avalible to convert a scipy.csr"));
#undef MAKE_NYMPY_FROM_HOMOGEN
    }
    else {
        throw std::invalid_argument("Output oneDAL table doesn't have homogen or csr format");
    }
    return res;
}

} // namespace oneapi::dal::python
