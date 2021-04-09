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

#include <cstdint>
#include <cstring>
#include <Python.h>
#include "data/backend/data.h"
#include "data/backend/utils.h"

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#ifdef ONEDAL_DATA_PARALLEL
    #include <CL/sycl.hpp>
#endif

#ifdef DPCTL_ENABLE
    #include "dpctl_sycl_types.h"
    #include "dpctl_sycl_queue_manager.h"
#endif

namespace oneapi::dal::python
{
#define is_array(a)           ((a) && PyArray_Check(a))
#define array_type(a)         PyArray_TYPE((PyArrayObject *)a)
#define array_is_behaved(a)   (PyArray_ISCARRAY_RO((PyArrayObject *)a) && array_type(a) < NPY_OBJECT)
#define array_is_behaved_F(a) (PyArray_ISFARRAY_RO((PyArrayObject *)a) && array_type(a) < NPY_OBJECT)
#define array_is_native(a)    (PyArray_ISNOTSWAPPED((PyArrayObject *)a))
#define array_numdims(a)      PyArray_NDIM((PyArrayObject *)a)
#define array_data(a)         PyArray_DATA((PyArrayObject *)a)
#define array_size(a, i)      PyArray_DIM((PyArrayObject *)a, i)

#ifdef ONEDAL_DATA_PARALLEL
int init_numpy()
{
    import_array();
    return 0;
}

const static int numpy_initialized = init_numpy();
#endif

class NumpyDeleter
{
public:
    NumpyDeleter(PyArrayObject * a) : _ndarray(a) {}

    void operator()(const void * ptr) {}

    NumpyDeleter & operator=(const NumpyDeleter &) = delete;

private:
    PyArrayObject * _ndarray;
};

template <typename T, typename ConstDeleter>
inline dal::homogen_table create_homogen_table(const T * data_pointer, const std::size_t row_count, const std::size_t column_count,
                                               const dal::data_layout layout, ConstDeleter && data_deleter)
{
    return dal::homogen_table(data_pointer, row_count, column_count, data_deleter, layout);
}

template <typename T>
inline dal::homogen_table make_homogen(PyArrayObject * array)
{
    size_t column_count = 1;

    if (array_numdims(array) > 2)
    {
        throw std::runtime_error("Input array has wrong dimensionality (must be 2d).");
    }
    T * data_pointer       = reinterpret_cast<T *>(array_data(array));
    const size_t row_count = static_cast<size_t>(array_size(array, 0));
    if (array_numdims(array) == 2)
    {
        column_count = static_cast<size_t>(array_size(array, 1));
    }
    const auto layout = array_is_behaved_F(array) ? dal::data_layout::column_major : dal::data_layout::row_major;
    auto res_table    = create_homogen_table(data_pointer, row_count, column_count, layout, NumpyDeleter(array));
    // we need it increment the ref-count if we use the input array in-place
    // if we copied/converted it we already own our own reference
    if (reinterpret_cast<PyArrayObject *>(data_pointer) == array) Py_INCREF(array);
    return res_table;
}

dal::table _input_to_onedal_table(PyObject * obj)
{
    dal::table res;
    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }
    if (obj == nullptr || obj == Py_None)
    {
        return res;
    }
    if (is_array(obj))
    {
        PyArrayObject * ary = reinterpret_cast<PyArrayObject *>(obj);
        if (array_is_behaved(ary) || array_is_behaved_F(ary))
        {
#define MAKE_HOMOGEN_TABLE(CType) res = make_homogen<CType>(ary);
            SET_NPY_FEATURE(PyArray_DESCR(ary)->type, MAKE_HOMOGEN_TABLE, throw std::invalid_argument("Found unsupported array type"));
#undef MAKE_HOMOGEN_TABLE
        }
        else
        {
            throw std::invalid_argument("[_input_to_onedal_table] Numpy input Could not convert Python object to onedal table.");
        }
    }
    else
    {
        throw std::invalid_argument("[_input_to_onedal_table] Not avalible input format for convert Python object to onedal table.");
    }
    return res;
}

class VSP
{
public:
    // we need a virtual destructor
    virtual ~VSP() {};
};

// typed virtual shared pointer, for simplicity we make it a oneDAL shared pointer
template <typename T>
class TVSP : public VSP, public dal::array<T>
{
public:
    TVSP(const dal::array<T> & org) : dal::array<T>(org) {}
    virtual ~TVSP() {};
};

void onedal_free_cap(PyObject * cap)
{
    VSP * stored_array = static_cast<VSP *>(PyCapsule_GetPointer(cap, NULL));
    if (stored_array)
    {
        delete stored_array;
    }
}

template <typename CType, int NpType>
static PyObject * convert_array_to_numpy(dal::array<CType> & array, std::int64_t row_count, std::int64_t column_count)
{
    npy_intp dims[2] = { static_cast<npy_intp>(row_count), static_cast<npy_intp>(column_count) };
    array.need_mutable_data();
    PyObject * obj = PyArray_SimpleNewFromData(2, dims, NpType, static_cast<void *>(array.get_mutable_data()));
    if (!obj) throw std::invalid_argument("Conversion to numpy array failed");

    void * opaque_value = static_cast<void *>(new TVSP<CType>(array));
    PyObject * cap      = PyCapsule_New(opaque_value, NULL, onedal_free_cap);
    PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(obj), cap);
    return obj;
}

PyObject * _table_to_numpy(const dal::table & input)
{
    PyObject * res = nullptr;
    if (!input.has_data())
    {
        throw std::invalid_argument("Empty data");
    }
    if (input.get_kind() == dal::homogen_table::kind())
    {
        const auto & homogen_res = static_cast<const dal::homogen_table &>(input);
        if (homogen_res.get_data_layout() == dal::data_layout::row_major)
        {
            const dal::data_type dtype = homogen_res.get_metadata().get_data_type(0);

#define MAKE_NYMPY_FROM_HOMOGEN(CType, NpType)                                                                                \
    {                                                                                                                         \
        auto rows = dal::row_accessor<const CType> { homogen_res }.pull();                                                    \
        res       = convert_array_to_numpy<CType, NpType>(rows, homogen_res.get_row_count(), homogen_res.get_column_count()); \
    }
            SET_CTYPE_NPY_FROM_DAL_TYPE(dtype, MAKE_NYMPY_FROM_HOMOGEN, throw std::invalid_argument("Not avalible to convert a numpy"));
#undef MAKE_NYMPY_FROM_HOMOGEN
        }
        else
        {
            throw std::invalid_argument("oneDAL have don't table row major format");
        }
    }
    else
    {
        throw std::invalid_argument("oneDAL table not homogen format have");
    }
    return res;
}

} // namespace oneapi::dal::python
