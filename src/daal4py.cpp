/*******************************************************************************
 * Copyright 2014-2019 Intel Corporation
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
#include <cstdint>
#include <cstring>
#include <Python.h>
#include "daal4py.h"
#include "npy4daal.h"


// ************************************************************************************
// ************************************************************************************
// Numpy type conversion code, taken from numpy.i (SWIG typemap-code)
// ************************************************************************************
// ************************************************************************************

#define is_array(a)            ((a) && PyArray_Check(a))
#define array_type(a)          PyArray_TYPE((PyArrayObject*)a)
#define array_is_behaved(a)    (PyArray_ISCARRAY_RO((PyArrayObject*)a) && array_type(a)<NPY_OBJECT)
#define array_is_behaved_F(a)  (PyArray_ISFARRAY_RO((PyArrayObject*)a) && array_type(a)<NPY_OBJECT)
#define array_is_native(a)     (PyArray_ISNOTSWAPPED((PyArrayObject*)a))
#define array_numdims(a)       PyArray_NDIM((PyArrayObject*)a)
#define array_data(a)          PyArray_DATA((PyArrayObject*)a)
#define array_size(a,i)        PyArray_DIM((PyArrayObject*)a,i)

// ************************************************************************************
// ************************************************************************************

class NumpyDeleter : public daal::services::DeleterIface
{
public:
    // constructor to initialize with ndarray
    NumpyDeleter(PyArrayObject* a) : _ndarray(a) {}
    // DeleterIface must be copy-constructible
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
    // We don't want this to be copied
    NumpyDeleter& operator=(const NumpyDeleter&) = delete;
private:
    PyArrayObject* _ndarray;
};

// define our own free functions for wrapping python objects holding our shared pointers
void daalsp_free_cap(PyObject * cap)
{
    VSP * sp = (VSP*) PyCapsule_GetPointer(cap, NULL);
    if (sp) delete sp;
}

// define our own free functions for wrapping python objects holding our raw pointers
void rawp_free_cap(PyObject * cap)
{
    void * rp = PyCapsule_GetPointer(cap, NULL);
    if (rp) delete[] rp;
}


void set_rawp_base(PyArrayObject * ary, void * ptr)
{
    PyObject* cap = PyCapsule_New(ptr, NULL, rawp_free_cap);
    PyArray_SetBaseObject(ary, cap);
}


// *****************************************************************************

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

#ifdef _DPCPP_
#include "oneapi/oneapi_api.h"
static int _1api_imp = import__oneapi();
#else
static int _1api_imp = -1;
PyObject* make_py_from_sycltable(void * ptr, int typ, int d1, int d2){ return Py_None; }
#endif
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
        if(_1api_imp == 0 && (res = make_py_from_sycltable(ptr, NPY_FLOAT64, (*ptr)->getNumberOfRows(), (*ptr)->getNumberOfColumns())) != Py_None) return res;
        if((res = _make_nda_from_bd<double, NPY_FLOAT64>(ptr)) != NULL) return res;
        break;
    case daal::data_management::data_feature_utils::DAAL_FLOAT32:
        if((res = _make_nda_from_homogen<float, NPY_FLOAT32>(ptr)) != NULL) return res;
        if(_1api_imp == 0 && (res = make_py_from_sycltable(ptr, NPY_FLOAT32, (*ptr)->getNumberOfRows(), (*ptr)->getNumberOfColumns())) != Py_None) return res;
        if((res = _make_nda_from_bd<float, NPY_FLOAT32>(ptr)) != NULL) return res;
        break;
    case daal::data_management::data_feature_utils::DAAL_INT32_S:
        if((res = _make_nda_from_homogen<int32_t, NPY_INT32>  (ptr)) != NULL) return res;
        if(_1api_imp == 0 && (res = make_py_from_sycltable(ptr, NPY_INT32, (*ptr)->getNumberOfRows(), (*ptr)->getNumberOfColumns())) != Py_None) return res;
        if((res = _make_nda_from_bd<int32_t, NPY_INT32>(ptr)) != NULL) return res;
        break;
    case daal::data_management::data_feature_utils::DAAL_INT32_U:
        if((res = _make_nda_from_homogen<uint32_t, NPY_UINT32> (ptr)) != NULL) return res;
        if(_1api_imp == 0 && (res = make_py_from_sycltable(ptr, NPY_UINT32, (*ptr)->getNumberOfRows(), (*ptr)->getNumberOfColumns())) != Py_None) return res;
        break;
    case daal::data_management::data_feature_utils::DAAL_INT64_S:
        if((res = _make_nda_from_homogen<int64_t, NPY_INT64>  (ptr)) != NULL) return res;
        if(_1api_imp == 0 && (res = make_py_from_sycltable(ptr, NPY_INT64, (*ptr)->getNumberOfRows(), (*ptr)->getNumberOfColumns())) != Py_None) return res;
        break;
    case daal::data_management::data_feature_utils::DAAL_INT64_U:
        if((res = _make_nda_from_homogen<uint64_t, NPY_UINT64> (ptr)) != NULL) return res;
        if(_1api_imp == 0 && (res = make_py_from_sycltable(ptr, NPY_UINT64, (*ptr)->getNumberOfRows(), (*ptr)->getNumberOfColumns())) != Py_None) return res;
        break;
    }
    // Falling back to using block-desriptors and converting to double
    if((res = _make_nda_from_bd<double, NPY_FLOAT64>(ptr)) != NULL) return res;

    throw std::invalid_argument("Got unsupported table type.");
}

// Create a list of numpy arrays
extern PyObject * make_nda(daal::data_management::DataCollectionPtr * coll)
{
    if(PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
    if(!coll) return Py_None;
    auto n = (*coll)->size();
    PyObject *list = PyList_New(n);
    for(auto i=0; i<n; ++i) {
        daal::data_management::NumericTablePtr nt = daal::services::dynamicPointerCast<daal::data_management::NumericTable>((*coll)->get(i));
        PyList_SetItem(list, i, make_nda(&nt));
        if(PyErr_Occurred()) { PyErr_Print(); throw std::runtime_error("Python error"); }
    }
    return list;
}

extern PyObject * make_nda(daal::data_management::KeyValueDataCollectionPtr * dict, const i2str_map_t & id2str)
{
    PyObject *pydict = PyDict_New();
    for(size_t i=0; i<(*dict)->size(); ++i) {
        auto elem = (*dict)->getValueByIndex(i);
        auto tbl = daal::services::dynamicPointerCast<daal::data_management::NumericTable>(elem);
        // There can be NULL elements in collection
        if(tbl || !elem) {
            PyObject * obj = tbl ? make_nda(&tbl) : Py_None;
            size_t key = (*dict)->getKeyByIndex(i);
            auto strkey = id2str.find(key);
            if(strkey != id2str.end()) {
                PyObject * keyobj = PyString_FromString(strkey->second.c_str());
                PyDict_SetItem(pydict, keyobj, obj);
                Py_DECREF(keyobj);
            } else {
                Py_DECREF(pydict);
                std::cerr << "Unexpected key '" << key << "' found in KeyValueDataCollectionPtr\n";
                return Py_None;
            }
        } else {
            Py_DECREF(pydict);
            std::cerr << "Unexpected object found in KeyValueDataCollectionPtr, expected NULL or NumericTable\n";
            return Py_None;
        }
    }
    return pydict;
}

template<typename T>
static daal::data_management::NumericTablePtr _make_hnt(PyObject * nda)
{
    daal::data_management::NumericTablePtr ptr;
    PyArrayObject * array = (PyArrayObject*)nda;

    assert(is_array(nda) && array_is_behaved(array));

    if(array_numdims(array) == 2) {
        // we provide the SharedPtr with a deleter which decrements the pyref
        ptr = daal::data_management::HomogenNumericTable<T>::create(daal::services::SharedPtr<T>((T*)array_data(array),
                                                                                                 NumpyDeleter(array)),
                                                                    (size_t)array_size(array,1),
                                                                    (size_t)array_size(array,0));
        // we need it increment the ref-count if we use the input array in-place
        // if we copied/converted it we already own our own reference
        if((PyObject*)array == nda) Py_INCREF(array);
    } else {
        std::cerr << "Input array has wrong dimensionality (must be 2d).\n";
    }

    return ptr;
}

 static daal::data_management::NumericTablePtr _make_npynt(PyObject * nda)
 {
    daal::data_management::NumericTable * ptr = NULL;

    assert(is_array(nda));

    PyArrayObject * array = (PyArrayObject*)nda;
    if(array_numdims(array) == 2) {
        // the given numpy array is not well behaved C array but has right dimensionality
        try {
            ptr = new NpyNumericTable<NpyNonContigHandler>(array);
        } catch (...) {
            ptr = NULL;
        }
    } else if(array_numdims(nda) == 1) {
        PyArray_Descr * descr = PyArray_DESCR(array);
        if(descr->names) {
            // the given array is a structured numpy array.
            ptr = new NpyNumericTable<NpyStructHandler>(array);
        } else {
            std::cerr << "Input array is neither well behaved and nor a structured array.\n";
        }
    } else {
        std::cerr << "Input array has wrong dimensionality (must be 2d).\n";
    }

    return daal::data_management::NumericTablePtr(ptr);
}

// Try to convert given object to DAAL Table without copying. Currently supports
// * numpy contiguous, homogenous -> DAAL HomogenNumericTable
// * numpy non-contiguous, homogenous -> NpyNumericTable
// * numpy structured, heterogenous -> NpyNumericTable
// * list of arrays, heterogen -> DAAL SOANumericTable
// * scipy csr_matrix -> DAAL CSRNumericTable
//   As long as DAAL CSR is only 0-based we need to copy indices/offsets
daal::data_management::NumericTablePtr make_nt(PyObject * obj)
{
    if(PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
    if(obj && obj != Py_None) {
	if(PyObject_HasAttrString(obj, "__2daalnt__")) {
            static daal::data_management::NumericTablePtr ntptr;
            if(true || !ntptr) {
            // special protocol assumes that python objects implement __2daalnt__
            // returning a pointer to a NumericTablePtr, we have to delete the shared-pointer
            PyObject * _obj = PyObject_CallMethod(obj, "__2daalnt__", NULL);
            if(PyErr_Occurred()) {PyErr_Print(); throw std::runtime_error("Python Error");}
            void * _ptr = PyCapsule_GetPointer(_obj, NULL);
            if(PyErr_Occurred()) {PyErr_Print(); throw std::runtime_error("Python Error");}
            Py_DECREF(_obj);
            auto nt = reinterpret_cast<daal::data_management::NumericTablePtr*>(_ptr);
            ntptr = *nt;
            delete nt; // we delete the shared pointer-pointer
            }

            return ntptr;
        }

        daal::data_management::NumericTablePtr ptr;
        if(is_array(obj)) { // we got a numpy array
            PyArrayObject * ary = (PyArrayObject*)obj;

            if(array_is_behaved(ary)) {
#define MAKENT_(_T) ptr = _make_hnt<_T>(obj)
                SET_NPY_FEATURE(PyArray_DESCR(ary)->type, MAKENT_, throw std::invalid_argument("Found unsupported array type"));
#undef MAKENT_
            } else {
		if(array_is_behaved_F(ary) && (PyArray_NDIM(ary) == 2)) {
		    int _axes = 0;
		    npy_intp N = PyArray_DIM(ary, 1); // number of columns
		    npy_intp column_len = PyArray_DIM(ary, 0);
		    int ary_numtype = PyArray_TYPE(ary);
		    /*
		     * Input is 2D F-contiguous array: represent it as SOA numeric table
		     */
		    daal::data_management::SOANumericTablePtr soatbl;

		    // iterate over columns
		    PyArrayIterObject * it = (PyArrayIterObject *) PyArray_IterAllButAxis(obj, &_axes);
		    if (it == NULL) {
                Py_XDECREF(it);
                throw std::runtime_error("Creating DAAL SOA table from F-contigous NumPy array failed: iterator could not be created");
		    }

		    soatbl = daal::data_management::SOANumericTable::create(N, column_len);

		    for(npy_intp i = 0; PyArray_ITER_NOTDONE(it); ++i) {
			PyArrayObject *slice = (PyArrayObject *) PyArray_SimpleNewFromData(1, &column_len, ary_numtype, (void *)PyArray_ITER_DATA(it));
			PyArray_SetBaseObject(slice, (PyObject *) ary);
			Py_INCREF(ary);
#define SETARRAY_(_T) {daal::services::SharedPtr< _T > _tmp(reinterpret_cast< _T * >(PyArray_ITER_DATA(it)), NumpyDeleter(slice)); soatbl->setArray(_tmp, i);}
			SET_NPY_FEATURE(PyArray_DESCR(ary)->type, SETARRAY_, throw std::invalid_argument("Found unsupported array type"));
#undef SETARRAY_
			PyArray_ITER_NEXT(it);
		    }

		    if(soatbl->getNumberOfColumns() != N) {
                throw std::runtime_error("Creating DAAL SOA table from F-contigous NumPy array failed.");
		    }
		    ptr = soatbl;
		} else
		    ptr = _make_npynt(obj);
            }

            if(!ptr) std::cerr << "Could not convert Python object to DAAL table.\n";

        } else if(PyList_Check(obj) && PyList_Size(obj) > 0) { // a list of arrays for SOA?
            PyObject * first = PyList_GetItem(obj, 0);
            if(PyErr_Occurred()) {PyErr_Print(); throw std::runtime_error("Python Error");}

            if(is_array(first)) { // can handle only list of 1d arrays
                auto N = PyList_Size(obj);
                daal::data_management::SOANumericTablePtr soatbl;

                for(auto i = 0; i < N; i++) {
                    PyArrayObject * ary = (PyArrayObject*)PyList_GetItem(obj, i);
                    if(PyErr_Occurred()) {PyErr_Print(); throw std::runtime_error("Python Error");}
                    if(i==0) soatbl = daal::data_management::SOANumericTable::create(N, PyArray_DIM(ary, 0));
                    if(PyArray_NDIM(ary) != 1) {
                        std::cerr << "Found wrong dimensionality (" << PyArray_NDIM(ary) << ") of array in list when constructing SOA table (must be 1d)";
                        break;
                    }

                    if (!array_is_behaved(ary))
                    {
                        std::cerr << "Can't get the data correctly without a copy.\n";
                    }

#define SETARRAY_(_T) {daal::services::SharedPtr< _T > _tmp(reinterpret_cast< _T * >(PyArray_DATA(ary)), NumpyDeleter(ary)); soatbl->setArray(_tmp, i);}
                    SET_NPY_FEATURE(PyArray_DESCR(ary)->type, SETARRAY_, throw std::invalid_argument("Found unsupported array type"));
#undef SETARRAY_
                    Py_INCREF(ary);
                }
                if(soatbl->getNumberOfColumns() != N) {
                    throw std::runtime_error("Creating DAAL SOA table from list failed.");
                }
                ptr = soatbl;
            } // else not a list of 1d arrays
        } // else not a list of 1d arrays
        if(!ptr && strcmp(Py_TYPE(obj)->tp_name, "csr_matrix") == 0) {
            daal::services::SharedPtr<daal::data_management::CSRNumericTable> ret;
            PyObject * vals  = PyObject_GetAttrString(obj, "data");
            if(PyErr_Occurred()) {PyErr_Print(); throw std::runtime_error("Python Error");}
            PyObject * indcs = PyObject_GetAttrString(obj, "indices");
            if(PyErr_Occurred()) {PyErr_Print(); throw std::runtime_error("Python Error");}
            PyObject * roffs = PyObject_GetAttrString(obj, "indptr");
            if(PyErr_Occurred()) {PyErr_Print(); throw std::runtime_error("Python Error");}
            PyObject * shape = PyObject_GetAttrString(obj, "shape");
            if(PyErr_Occurred()) {PyErr_Print(); throw std::runtime_error("Python Error");}

            if(shape && PyTuple_Check(shape)
               && is_array(vals) && is_array(indcs) && is_array(roffs)
               && array_numdims(vals)==1 && array_numdims(indcs)==1 && array_numdims(roffs)==1) {
                if(PyErr_Occurred()) {PyErr_Print(); throw std::runtime_error("Python Error");}

                // As long as DAAL does not support 0-based indexing we have to copy the indices and add 1 to each
                PyObject * np_indcs = PyArray_FROMANY(indcs, NPY_UINT64, 0, 0, NPY_ARRAY_CARRAY|NPY_ARRAY_ENSURECOPY|NPY_ARRAY_FORCECAST);
                if(PyErr_Occurred()) {PyErr_Print(); throw std::runtime_error("Python Error");}
                PyObject * np_roffs = PyArray_FROMANY(roffs, NPY_UINT64, 0, 0, NPY_ARRAY_CARRAY|NPY_ARRAY_ENSURECOPY|NPY_ARRAY_FORCECAST);
                if(PyErr_Occurred()) {PyErr_Print(); throw std::runtime_error("Python Error");}
                PyObject * np_vals = PyArray_FROMANY(vals, array_type(vals), 0, 0, NPY_ARRAY_CARRAY);
                if(PyErr_Occurred()) {PyErr_Print(); throw std::runtime_error("Python Error");}

                PyObject * nr = PyTuple_GetItem(shape, 0);
                if(PyErr_Occurred()) {PyErr_Print(); throw std::runtime_error("Python Error");}
                PyObject * nc = PyTuple_GetItem(shape, 1);
                if(PyErr_Occurred()) {PyErr_Print(); throw std::runtime_error("Python Error");}

                if(np_indcs && np_roffs && np_vals && nr && nc) {
                    // for now, increment indcs by 1
                    size_t * c_indcs = (size_t*)array_data(np_indcs);
                    size_t n = array_size(np_indcs, 0);
                    for(size_t i=0; i<n; ++i) c_indcs[i] += 1;

                    size_t * c_roffs = (size_t*)array_data(np_roffs);
                    n = array_size(np_roffs, 0);
                    for(size_t i=0; i<n; ++i) c_roffs[i] += 1;

                    size_t c_nc = (size_t)PyInt_AsSsize_t(nc);
                    if(PyErr_Occurred()) {PyErr_Print(); throw std::runtime_error("Python Error");}
                    size_t c_nr = (size_t)PyInt_AsSsize_t(nr);
                    if(PyErr_Occurred()) {PyErr_Print(); throw std::runtime_error("Python Error");}

#define MKCSR_(_T)                                                      \
                    ret = daal::data_management::CSRNumericTable::create(daal::services::SharedPtr<_T>((_T*)array_data(np_vals), NumpyDeleter((PyArrayObject*)np_vals)), \
                                                                         daal::services::SharedPtr<size_t>(c_indcs, NumpyDeleter((PyArrayObject*)np_indcs)), \
                                                                         daal::services::SharedPtr<size_t>(c_roffs, NumpyDeleter((PyArrayObject*)np_roffs)), \
                                                                         c_nc, \
                                                                         c_nr)
                    SET_NPY_FEATURE(array_type(np_vals), MKCSR_, throw std::invalid_argument("Found unsupported data type in csr_matrix"));
#undef MKCSR_

                } else std::cerr << "Failed accessing csr data when converting csr_matrix.\n";
            } else std::cerr << "Got invalid csr_matrix object.\n";
            Py_DECREF(shape);
            Py_DECREF(roffs);
            Py_DECREF(indcs);
            Py_DECREF(vals);
            return daal::data_management::NumericTablePtr(ret);
        }

        return ptr;
    }

    return daal::data_management::NumericTablePtr();
}

extern daal::data_management::KeyValueDataCollectionPtr make_dnt(PyObject * dict, str2i_map_t & str2id)
{
    daal::data_management::KeyValueDataCollectionPtr dc(new daal::data_management::KeyValueDataCollection);
    if(dict && dict != Py_None) {
        if(PyDict_Check(dict)) {
            PyObject *key, *value;
            Py_ssize_t pos = 0;
            while(PyDict_Next(dict, &pos, &key, &value)) {
                const char *strkey = PyString_AsString(key);
                auto keyid = str2id.find(strkey);
                if(keyid != str2id.end()) {
                    daal::data_management::NumericTablePtr tbl = make_nt(value);
                    if(tbl) {
                        (*dc)[keyid->second] = daal::services::staticPointerCast<daal::data_management::SerializationIface>(tbl);
                    } else {
                        std::cerr << "Unexpected object '" << Py_TYPE(value)->tp_name << "' found in dict, expected an array\n";
                    }
                } else {
                    std::cerr << "Unexpected key '" << Py_TYPE(key)->tp_name << "' found in dict, expected a string\n";
                }
            }
        } else {
            std::cerr << "Unexpected object '" << Py_TYPE(dict)->tp_name << "' found, expected dict\n";
        }
    }
    return dc;
}

data_or_file::data_or_file(PyObject * input)
{
    this->table.reset();
    this->file.resize(0);
    if(input == Py_None) {
        ;
    } else if(PyUnicode_Check(input)) {
        //        this->file = PyUnicode_AsUTF8AndSize(input, &size);
        this->file = PyUnicode_AsUTF8(input);
    } else {
        auto tmp = make_nt(input);
        if(tmp) {
            this->table = tmp;
        }
        if(! this->table) {
            std::cerr << "Got type '" << Py_TYPE(input)->tp_name << "' when expecting string, array, or list of 1d-arrays. Treating as None." << std::endl;
        }
    }
}

const daal::data_management::NumericTablePtr get_table(const data_or_file & t)
{
    if(t.table) return t.table;
    if(t.file.size()) return readCSV(t.file);
    throw std::invalid_argument("one and only one input per process allowed");
    return daal::data_management::NumericTablePtr();
}

const daal::data_management::NumericTablePtr readCSV(const std::string& fname)
{
    daal::data_management::FileDataSource< daal::data_management::CSVFeatureManager >
        dataSource(fname,
                   daal::data_management::DataSource::doAllocateNumericTable,
                   daal::data_management::DataSource::doDictionaryFromContext);
    dataSource.loadDataBlock();
    return dataSource.getNumericTable();
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


daal::data_management::DataCollectionPtr make_datacoll(PyObject * input)
{
    if(PyErr_Occurred()) {PyErr_Print(); PyErr_Clear();}
    if(input && input != Py_None && PyList_Check(input) && PyList_Size(input) > 0) {
        auto n = PyList_Size(input);
        daal::data_management::DataCollection * res = new daal::data_management::DataCollection;
        res->resize(n);
        for(auto i = 0; i < n; i++) {
            PyObject * obj = PyList_GetItem(input, i);
            if(PyErr_Occurred()) {PyErr_Print(); throw std::runtime_error("Python Error");}
            auto tmp = make_nt(obj);
            if(tmp) res->push_back(tmp);
            else throw std::runtime_error(std::string("Unexpected object '") + Py_TYPE(obj)->tp_name + "' found in list, expected an array");
        }
        return daal::data_management::DataCollectionPtr(res);
    }
    return daal::data_management::DataCollectionPtr();
}


static int64_t getval_(const std::string& str, const str2i_map_t & strmap)
{
        auto i = strmap.find(str);
        if(i == strmap.end()) throw std::invalid_argument(std::string("Encountered unexpected string-identifier '")
                                                                      + str
                                                                      + std::string("'"));
        return i->second;
}

int64_t string2enum(const std::string& str, str2i_map_t & strmap)
{
    int64_t r = 0;
    std::size_t current, previous = 0;
    while((current = str.find('|', previous)) != std::string::npos) {
        r |= getval_(str.substr(previous, current - previous), strmap);
        previous = current + 1;
    }
    return (r | getval_(str.substr(previous, current - previous), strmap));
}


#ifdef _DIST_
#include "transceiver.h"
#endif

extern "C" {
void c_daalinit(int nthreads)
{
    if(nthreads > 0) daal::services::Environment::getInstance()->setNumberOfThreads(nthreads);
}

void c_daalfini()
{
#ifdef _DIST_
    del_transceiver();
#endif
}

size_t c_num_threads()
{
    return daal::services::Environment::getInstance()->getNumberOfThreads();
}


size_t c_num_procs()
{
#ifdef _DIST_
    return get_transceiver()->nMembers();
#else
    return 1;
#endif
}

size_t c_my_procid()
{
#ifdef _DIST_
    return get_transceiver()->me();
#else
    return 0;
#endif
}
} // extern "C"
