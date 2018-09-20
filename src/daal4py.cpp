/*******************************************************************************
 * Copyright 2014-2018 Intel Corporation
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

#define NO_IMPORT_ARRAY
#include "daal4py.h"
#include <cstdint>


// ************************************************************************************
// ************************************************************************************
// Numpy type conversion code, taken from numpy.i (SWIG typemap-code)
// ************************************************************************************
// ************************************************************************************

#define is_array(a)            ((a) && PyArray_Check(a))
#define array_is_contiguous(a) (PyArray_ISCONTIGUOUS((PyArrayObject*)a))
#define array_is_native(a)     (PyArray_ISNOTSWAPPED((PyArrayObject*)a))
#define array_numdims(a)       PyArray_NDIM((PyArrayObject*)a)
#define array_type(a)          PyArray_TYPE((PyArrayObject*)a)
#define array_data(a)          PyArray_DATA((PyArrayObject*)a)
#define array_size(a,i)        PyArray_DIM((PyArrayObject*)a,i)

/* Test whether a python object is contiguous.  If array is
 * contiguous, return 1.  Otherwise, set the python error string and
 * return 0.
 */
int require_contiguous(PyArrayObject* ary)
{
    int contiguous = 1;
    if (!array_is_contiguous(ary)) {
        PyErr_SetString(PyExc_TypeError,
                        "Array must be contiguous.  A non-contiguous array was given");
        contiguous = 0;
    }
    return contiguous;
}

/* Require that a numpy array is not byte-swapped.  If the array is
 * not byte-swapped, return 1.  Otherwise, set the python error string
 * and return 0.
 */
int require_native(PyArrayObject* ary)
{
    int native = 1;
    if (!array_is_native(ary)) {
        PyErr_SetString(PyExc_TypeError,
                        "Array must have native byteorder.  "
                        "A byte-swapped array was given");
        native = 0;
    }
    return native;
}

/* Require the given PyArrayObject to have a specified number of
 * dimensions.  If the array has the specified number of dimensions,
 * return 1.  Otherwise, set the python error string and return 0.
 */
int require_dimensions(PyArrayObject* ary,
                       int            exact_dimensions)
{
    int success = 1;
    if (array_numdims(ary) != exact_dimensions) {
        PyErr_Format(PyExc_TypeError,
                     "Array must have %d dimensions.  Given array has %d dimensions",
                     exact_dimensions,
                     array_numdims(ary));
        success = 0;
    }
    return success;
}

/* Given a PyArrayObject, check to see if it is contiguous.  If so,
 * return the input pointer and flag it as not a new object.  If it is
 * not contiguous, create a new PyArrayObject using the original data,
 * flag it as a new object and return the pointer.
 */
PyArrayObject* make_contiguous(PyArrayObject* ary,
                               int*           is_new_object,
                               int            min_dims,
                               int            max_dims)
{
    PyArrayObject* result;
    if (array_is_contiguous(ary)) {
        result = ary;
        *is_new_object = 0;
    } else {
        result = (PyArrayObject*) PyArray_ContiguousFromObject((PyObject*)ary,
                                                               array_type(ary),
                                                               min_dims,
                                                               max_dims);
        *is_new_object = 1;
    }
    return result;
}

/* Convert the given PyObject to a NumPy array with the given
 * typecode.  On success, return a valid PyArrayObject* with the
 * correct type.  On failure, the python error string will be set and
 * the routine returns NULL.
 */
PyArrayObject* obj_to_array_allow_conversion(PyObject* input,
                                             int       typecode,
                                             int*      is_new_object)
{
    PyArrayObject* ary = NULL;
    PyObject*      py_obj;
    if (is_array(input) && (typecode == NPY_NOTYPE ||
                            PyArray_EquivTypenums(array_type(input),typecode))) {
        ary = (PyArrayObject*) input;
        *is_new_object = 0;
    } else {
        py_obj = PyArray_FROMANY(input, typecode, 0, 0, NPY_ARRAY_DEFAULT);
        /* If NULL, PyArray_FromObject will have set python error value.*/
        ary = (PyArrayObject*) py_obj;
        *is_new_object = 1;
    }
    return ary;
}

/* Convert a given PyObject to a contiguous PyArrayObject of the
 * specified type.  If the input object is not a contiguous
 * PyArrayObject, a new one will be created and the new object flag
 * will be set.
 */
PyArrayObject* obj_to_array_contiguous_allow_conversion(PyObject* input,
                                                        int       typecode,
                                                        int*      is_new_object)
{
    int is_new1 = 0;
    int is_new2 = 0;
    PyArrayObject* ary2;
    PyArrayObject* ary1 = obj_to_array_allow_conversion(input,
                                                        typecode,
                                                        &is_new1);
    if (ary1) {
        ary2 = make_contiguous(ary1, &is_new2, 0, 0);
        if ( is_new1 && is_new2) {
            Py_DECREF(ary1);
        }
        ary1 = ary2;
    }
    *is_new_object = is_new1 || is_new2;
    return ary1;
}

// ************************************************************************************
// ************************************************************************************

class NumpyDeleter : public daal::services::DeleterIface
{
public:
    // constructor to initialize with ndarray
    NumpyDeleter(PyArrayObject* a) : _ndarray(a) {}
    // DeleterIface must be copy-constrible
    NumpyDeleter(const NumpyDeleter & o) : _ndarray(o._ndarray) {}
    // ref-count reached 0 -> decref reference to python object
    void operator() (const void *ptr) DAAL_C11_OVERRIDE
    {
        // We need to protect calls to python API
        // Note: at termination time, even when no threads are running, this breaks without the protection
        PyGILState_STATE gstate = PyGILState_Ensure();
        assert((void *)array_data(_ndarray) == ptr);
        Py_DECREF(_ndarray);
        PyGILState_Release(gstate);
    }
private:
    PyArrayObject* _ndarray;
};

// An empty virtual base class (used by TVSP) for shared pointer handling
// we use this to have a generic type for all shared pointers
// e.g. used in daalsp_free functions below
class VSP 
{
public:
    // we need a virtual destructor
    virtual ~VSP() {};
};
// typed virtual shared pointer, for simplicity we make it a DAAL shared pointer
template< typename T >
class TVSP : public VSP, public daal::services::SharedPtr<T>
{
public:
    TVSP(const daal::services::SharedPtr<T> & org) : daal::services::SharedPtr<T>(org) {}
    virtual ~TVSP() {};
};

// define our own free functions for wrapping python objects holding our shared pointers
#ifdef USE_CAPSULE
void daalsp_free_cap(PyObject * cap)
{
    VSP * sp = (VSP*) PyCapsule_GetPointer(cap, NULL);
    if (sp) delete sp;
}
#else
void daalsp_free(void * cap)
{
    VSP * sp = (VSP*) PyCObject_AsVoidPtr(reinterpret_cast<PyObject *>(cap));
    if (sp) delete sp;
}
#endif
     
template< typename T >
void set_sp_base(PyArrayObject * ary, daal::services::SharedPtr<T> & sp)
{
    void * tmp_sp = (void*) new TVSP<T>(sp);
#ifdef USE_CAPSULE
    PyObject* cap = PyCapsule_New(tmp_sp, NULL, daalsp_free_cap);
#else
    PyObject* cap = PyCObject_FromVoidPtr(tmp_sp, daalsp_free);
#endif
    PyArray_SetBaseObject(ary, cap);
}


TableOrFList::TableOrFList(PyObject * input)
{
    this->table.reset();
    this->tlist.resize(0);
    this->file.resize(0);
    this->flist.resize(0);
    if(input == Py_None) {
        ;
    } else if(PyList_Check(input) && PyList_Size(input) > 0) {
        PyObject * first = PyList_GetItem(input, 0);
        if(is_array(first)) {
            this->tlist.resize(PyList_Size(input));
            for(auto i = 0; i < this->tlist.size(); i++) {
                int is_new_object = 0;
                PyObject * el = PyList_GetItem(input, i);
                PyArrayObject* array = obj_to_array_contiguous_allow_conversion(el, NPY_FLOAT64, &is_new_object);
                if (!array || !require_dimensions(array,2) || !require_contiguous(array) || !require_native(array)) {
                    throw std::invalid_argument("Array converstion failed.");
                }
                // we provide the SharedPtr with a deleter which decrements the pyref
                this->tlist[i].reset(new daal::data_management::HomogenNumericTable<double>(daal::services::SharedPtr<double>((double*)array_data(array),
                                                                                                                              NumpyDeleter(array)),
                                                                                            (size_t)array_size(array,1),
                                                                                            (size_t)array_size(array,0)));
                // we need it increment the ref-count if we use the input array in-place
                // if we copied/converted it we already own our own reference
                if((PyObject*)array == el) Py_INCREF(array);
            }
        } else if(PyUnicode_Check(first)) {
            this->flist.resize(PyList_Size(input));
            for(auto i = 0; i < this->flist.size(); i++) {
                this->flist[i] = PyUnicode_AsUTF8(PyList_GetItem(input, i));
            }
        }
    } else if(PyUnicode_Check(input)) {
        //        this->file = PyUnicode_AsUTF8AndSize(input, &size);
        this->file = PyUnicode_AsUTF8(input);
    } else if(is_array(input)) {
        int is_new_object = 0;
        PyArrayObject* array = obj_to_array_contiguous_allow_conversion(input, NPY_FLOAT64, &is_new_object);
        if (!array || !require_dimensions(array,2) || !require_contiguous(array) || !require_native(array)) {
            throw std::invalid_argument("Array converstion failed.");
            return;
        }
        // we provide the SharedPtr with a deleter which decrements the pyref
        this->table.reset(new daal::data_management::HomogenNumericTable<double>(daal::services::SharedPtr<double>((double*)array_data(array),
                                                                                                                   NumpyDeleter(array)),
                                                                                 (size_t)array_size(array,1),
                                                                                 (size_t)array_size(array,0)));
        // we need it increment the ref-count if we use the input array in-place
        // if we copied/converted it we already own our own reference
        if((PyObject*)array == input) Py_INCREF(array);
    } else {
        std::cerr << "Got type '" << Py_TYPE(input)->tp_name << "' when expecting string or array or list of strings/arrays. Treating as None." << std::endl;
    }
}


// Uses a shared pointer to a raw array (T*) for creating a nd-array
template<typename T, int NPTYPE>
static PyObject * _sp_to_nda(daal::services::SharedPtr< T > & sp, size_t nr, size_t nc)
{
    npy_intp dims[2] = {static_cast<npy_intp>(nr), static_cast<npy_intp>(nc)};
    PyObject* obj = PyArray_SimpleNewFromData(2, dims, NPTYPE, (void*)sp.get());
    if (!obj) throw std::invalid_argument("conversion to numpy array failed");
    set_sp_base((PyArrayObject*)obj, sp);
    return obj;
}

// get a block of Rows from NT and then craete nd-array from it
// DAAL potentially makes a copy when creating the BlockDesriptor
template<typename T, int NPTYPE>
static PyObject * _make_nda_from_bd(daal::data_management::NumericTablePtr * ptr)
{
    daal::data_management::BlockDescriptor<T> block;
    (*ptr)->getBlockOfRows(0, (*ptr)->getNumberOfRows(), daal::data_management::readOnly, block);
    if(block.getNumberOfRows() != (*ptr)->getNumberOfRows() || block.getNumberOfColumns() != (*ptr)->getNumberOfColumns()) {
        std::cerr << "Getting data ptr as block-of-rows failed.\nExpected shape: "
                  << (*ptr)->getNumberOfRows() << "x" << (*ptr)->getNumberOfColumns() << "\nBlock shape:"
                  << block.getNumberOfRows() << "x" <<  block.getNumberOfColumns() << std::endl;
        return NULL;
    }
    daal::services::SharedPtr< T > data_tmp = block.getBlockSharedPtr();
    if(! data_tmp) {
        std::cerr << "Unexpected null pointer from block descriptor.";
        return NULL;
    }
    return _sp_to_nda<T, NPTYPE>(data_tmp, block.getNumberOfRows(), block.getNumberOfColumns());
}

// Most efficient conversion if NT is a HomogeNumericTable
// We do not need to make a copy and can use the raw data pointer directly.
template<typename T, int NPTYPE>
static PyObject * _make_nda_from_homogen(daal::data_management::NumericTablePtr * ptr)
{
    auto dptr = dynamic_cast< daal::data_management::HomogenNumericTable< T >* >((*ptr).get());
    if(dptr) {
        daal::services::SharedPtr< T > data_tmp(dptr->getArraySharedPtr());
        return _sp_to_nda<T, NPTYPE>(data_tmp, (*ptr)->getNumberOfRows(), (*ptr)->getNumberOfColumns());
    }
    return NULL;
}

// Convert a DAAL NT to a numpy nd-array
// tries to avoid copying the data, instead we try to share the memory with DAAL
PyObject * make_nda(daal::data_management::NumericTablePtr * ptr)
{
    if(!ptr
       || !(*ptr).get()
       || (*ptr)->getNumberOfRows() == 0
       || (*ptr)->getNumberOfRows() == 0) return Py_None;

    PyObject * res = NULL;

    // Try to convert from homogen/dense type as given in first column of NT
    // first try HomogenNT, then via BlockDescriptor. The latter requires a copy.
    switch((*(*ptr)->getDictionary())[0].indexType) {
    case daal::data_management::data_feature_utils::DAAL_FLOAT64:
        if((res = _make_nda_from_homogen<double, NPY_FLOAT64>(ptr)) != NULL) return res;
        if((res = _make_nda_from_bd<double, NPY_FLOAT64>(ptr)) != NULL) return res;
        break;
    case daal::data_management::data_feature_utils::DAAL_FLOAT32:
        if((res = _make_nda_from_homogen<float, NPY_FLOAT32>(ptr)) != NULL) return res;
        if((res = _make_nda_from_bd<float, NPY_FLOAT32>(ptr)) != NULL) return res;
        break;
    case daal::data_management::data_feature_utils::DAAL_INT32_S:
        if((res = _make_nda_from_homogen<int32_t, NPY_INT32>  (ptr)) != NULL) return res;
        if((res = _make_nda_from_bd<int32_t, NPY_INT32>(ptr)) != NULL) return res;
        break;
    case daal::data_management::data_feature_utils::DAAL_INT32_U:
        if((res = _make_nda_from_homogen<uint32_t, NPY_UINT32> (ptr)) != NULL) return res;
        break;
    case daal::data_management::data_feature_utils::DAAL_INT64_S:
        if((res = _make_nda_from_homogen<int64_t, NPY_INT64>  (ptr)) != NULL) return res;
        break;
    case daal::data_management::data_feature_utils::DAAL_INT64_U:
        if((res = _make_nda_from_homogen<uint64_t, NPY_UINT64> (ptr)) != NULL) return res;
        break;
    }
    // Falling back to using block-desriptors and converting to double
    if((res = _make_nda_from_bd<double, NPY_FLOAT64>(ptr)) != NULL) return res;

    throw std::invalid_argument("Got unsupported table type.");
}

template<typename T, int NPTYPE>
static daal::data_management::NumericTable * _make_nt(PyObject * nda)
{
    int is_new_object = 0;
    PyArrayObject* array = obj_to_array_contiguous_allow_conversion(nda, NPTYPE, &is_new_object);
    if (!array || !require_dimensions(array,2) || !require_contiguous(array) || !require_native(array)) {
        throw std::invalid_argument("Conversion to DAAL NumericTable failed");
    }
    // we provide the SharedPtr with a deleter which decrements the pyref
    daal::data_management::NumericTable * ptr = new daal::data_management::HomogenNumericTable<T>(daal::services::SharedPtr<T>((T*)array_data(array),
                                                                                                                               NumpyDeleter(array)),
                                                                                                  (size_t)array_size(array,1),
                                                                                                  (size_t)array_size(array,0));
    // we need it increment the ref-count if we use the input array in-place
    // if we copied/converted it we already own our own reference
    if((PyObject*)array == nda) Py_INCREF(array);

    return ptr;
}

daal::data_management::NumericTablePtr * make_nt(PyObject * nda)
{
    if(nda && nda != Py_None) {
        daal::data_management::NumericTable * ptr = NULL;

        switch(array_type(nda)) {
        case NPY_UINT32:
            ptr = _make_nt<uint32_t, NPY_UINT32>(nda);
            break;
        case NPY_INT32:
            ptr = _make_nt<int32_t, NPY_INT32>(nda);
            break;
        case NPY_UINT64:
            ptr = _make_nt<uint64_t, NPY_UINT64>(nda);
            break;
        case NPY_INT64:
            ptr = _make_nt<int64_t, NPY_INT64>(nda);
            break;
        case NPY_FLOAT32:
            ptr = _make_nt<float, NPY_FLOAT32>(nda);
            break;
        default:
            ptr = _make_nt<double, NPY_FLOAT64>(nda);
        }

        return new daal::data_management::NumericTablePtr(ptr);
    }
	return new daal::data_management::NumericTablePtr();
}

const daal::data_management::NumericTablePtr readCSV(const std::string& fname)
{
    daal::data_management::FileDataSource< daal::data_management::CSVFeatureManager >
        dataSource(fname,
                   daal::data_management::DataSource::doAllocateNumericTable,
                   daal::data_management::DataSource::doDictionaryFromContext);
    dataSource.loadDataBlock();
    return daal::data_management::NumericTablePtr(dataSource.getNumericTable());
}


extern "C"
void to_c_array(const daal::data_management::NumericTablePtr * ptr, void ** data, size_t * dims, char dtype)
{
    *data = NULL;
    if(ptr && ptr->get()) {
        dims[0] = (*ptr)->getNumberOfRows();
        dims[1] = (*ptr)->getNumberOfColumns();
        switch(dtype) {
        case 0:
            *data = get_nt_data_ptr< double >(ptr);
            break;
        case 1:
            *data = get_nt_data_ptr< float >(ptr);
            break;
        case 2:
            *data = get_nt_data_ptr< int >(ptr);
            break;
        default:
			std::cerr << "Invalid data type specified." << std::endl;
        }
        if(*data) return;
		std::cerr << "Data type and table type are incompatible." << std::endl;
    }
    // ptr==NULL: no input data
    dims[0] = dims[1] = 0;
    return;
}

int64_t string2enum(const std::string& str, std::map< std::string, int64_t > & strmap)
{
    int64_t r = 0;
    std::size_t current, previous = 0;
    while((current = str.find('|', previous)) != std::string::npos) {
        r |= strmap[str.substr(previous, current - previous)];
        previous = current + 1;
    }
    return (r | strmap[str.substr(previous, current - previous)]);
}
