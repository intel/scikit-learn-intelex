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

#ifndef __NPY4DAAL_H_INCLUDED__
#define __NPY4DAAL_H_INCLUDED__

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#if PY_VERSION_HEX >= 0x03000000
#define PyString_Check(name) PyUnicode_Check(name)
#define PyString_AsString(str) PyUnicode_AsUTF8(str)
#define PyString_FromString(str) PyUnicode_FromString(str)
#define PyString_FromStringAndSize(str, sz) PyUnicode_FromStringAndSize(str, sz)
#endif


#define SET_NPY_FEATURE( _T, _M )                \
    switch(_T) { \
        case NPY_DOUBLE: \
        case NPY_CDOUBLE: \
        case NPY_DOUBLELTR: \
        case NPY_CDOUBLELTR: \
            _M(double); \
            break; \
        case NPY_FLOAT: \
        case NPY_CFLOAT: \
        case NPY_FLOATLTR: \
        case NPY_CFLOATLTR: \
            _M(float); \
            break; \
        case NPY_INT: \
        case NPY_INTLTR: \
            _M(int); \
            break; \
        case NPY_UINT: \
        case NPY_UINTLTR: \
            _M(unsigned int); \
            break; \
        case NPY_LONG: \
        case NPY_LONGLTR: \
            _M(long); \
            break; \
        case NPY_ULONG: \
        case NPY_ULONGLTR: \
            _M(unsigned long); \
            break; \
        case NPY_LONGLONG: \
        case NPY_LONGLONGLTR: \
            _M(long long); \
            break; \
        case NPY_ULONGLONG: \
        case NPY_ULONGLONGLTR: \
            _M(unsigned long long); \
            break; \
        case NPY_BYTE: \
        case NPY_BYTELTR: \
            _M(char);\
            break; \
        case NPY_UBYTE:  \
        case NPY_UBYTELTR:  \
            _M(unsigned char);\
            break; \
        case NPY_SHORT: \
        case NPY_SHORTLTR: \
            _M(short);\
            break; \
        case NPY_USHORT: \
        case NPY_USHORTLTR: \
            _M(unsigned short); \
            break; \
        default: \
            std::cerr << "Unsupported NPY type " << (_T) << " ignored\n."; \
            return; \
    };

template<typename T> struct npy_type;
template<> struct npy_type<double> { static constexpr char *value = "f8"; };
template<> struct npy_type<float>  { static constexpr char *value = "f4"; };
template<> struct npy_type<int>    { static constexpr char *value = "i4"; };


// Numeric Table wrapping a non-contiguous, homogen numpy array
// Avoids copying by using numpy iterators when accesing blocks of data
class NpyNumericTable : public daal::data_management::NumericTable
{
protected:
    PyArrayObject * _ary;
public:

    /**
     *  Constructor
     *  \param[in]  ary  The non-contiguous, homogen numpy array to wrap
     */
    NpyNumericTable(PyArrayObject * ary)
        : NumericTable(daal::data_management::NumericTableDictionaryPtr()),
          _ary(ary)
    {
        Py_XINCREF(ary);
        _layout = daal::data_management::NumericTableIface::aos;

        PyArray_Descr * descr = PyArray_DESCR(ary);              // type descriptor

        // we assume numpy.i has done typechecks and this is a 2-dimensional homogen array
        if(descr->names) {
            std::cerr << "Found a structured numpy array. Unable to create homogen NumericTable." << std::endl;
            this->_status.add(daal::services::ErrorIncorrectTypeOfInputNumericTable);
            return;
        }
        if(PyArray_NDIM(ary) != 2) {
            std::cerr << "Found array with " << PyArray_NDIM(ary) << " dimensions, extected 2. Don't know how to create homogen NumericTable." << std::endl;
            this->_status.add(daal::services::ErrorIncorrectTypeOfInputNumericTable);
            return;
        }

        Py_ssize_t N = PyArray_DIMS(ary)[1];
        _ddict = daal::data_management::NumericTableDictionaryPtr(new daal::data_management::NumericTableDictionary(N));
        setNumberOfRows(PyArray_DIMS(ary)[0]);
        // setNumberOfColumns not needed, done by providing size to ddict

        // iterate through all elements and init ddict feature accordingly
        for (Py_ssize_t i=0; i<N; i++) {
#define SETFEATURE_(_T) _ddict->setFeature<_T>(i)
            SET_NPY_FEATURE(descr->type, SETFEATURE_);
#undef SETFEATURE_
        }
        _memStatus = daal::data_management::NumericTableIface::userAllocated;
    }

    /** \private */
    ~NpyNumericTable()
    {
        Py_XDECREF(_ary);
    }

    virtual daal::services::Status resize(size_t nrows) DAAL_C11_OVERRIDE
    {
        std::cerr << "Resizing numpy array through daal not supported." << std::endl;
        return daal::services::Status(daal::services::ErrorMethodNotSupported);
    }

    virtual int getSerializationTag() const
    {
        return daal::SERIALIZATION_AOS_NT_ID;
    }

    daal::services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, daal::data_management::ReadWriteMode rwflag, daal::data_management::BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    daal::services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, daal::data_management::ReadWriteMode rwflag, daal::data_management::BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    daal::services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, daal::data_management::ReadWriteMode rwflag, daal::data_management::BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    daal::services::Status releaseBlockOfRows(daal::data_management::BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        return releaseTBlock<double>(block);
    }
    daal::services::Status releaseBlockOfRows(daal::data_management::BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        return releaseTBlock<float>(block);
    }
    daal::services::Status releaseBlockOfRows(daal::data_management::BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        return releaseTBlock<int>(block);
    }

    daal::services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, daal::data_management::ReadWriteMode rwflag,
                                            daal::data_management::BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<double>(vector_idx, value_num, rwflag, block, feature_idx, 1 );
    }
    daal::services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, daal::data_management::ReadWriteMode rwflag,
                                            daal::data_management::BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<float>(vector_idx, value_num, rwflag, block, feature_idx, 1);
    }
    daal::services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, daal::data_management::ReadWriteMode rwflag,
                                            daal::data_management::BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<int>(vector_idx, value_num, rwflag, block, feature_idx, 1);
    }

    daal::services::Status releaseBlockOfColumnValues(daal::data_management::BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        return releaseTBlock<double>(block);
    }
    daal::services::Status releaseBlockOfColumnValues(daal::data_management::BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        return releaseTBlock<float>(block);
    }
    daal::services::Status releaseBlockOfColumnValues(daal::data_management::BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        return releaseTBlock<int>(block);
    }

    daal::services::Status allocateDataMemory(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE
    {
        return daal::services::Status(daal::services::ErrorMethodNotSupported);
    }

    void freeDataMemory() DAAL_C11_OVERRIDE
    {
        daal::services::Status ec(daal::services::ErrorMethodNotSupported);
    }

    /** \private */
    daal::services::Status serializeImpl(daal::data_management::InputDataArchive  *archive)
    {
        // To make our lives easier, we first create a contiguous array
        PyArrayObject * ary = PyArray_GETCONTIGUOUS(_ary);
        // First serialize the type descriptor in string representation
        Py_ssize_t len = 0;
#if PY_MAJOR_VERSION < 3
        char * ds = NULL;
        PyString_AsStringAndSize(PyObject_Repr(PyArray_DESCR(ary)), &ds, &len);
#else
        char * ds = PyUnicode_AsUTF8AndSize(PyObject_Repr((PyObject*)PyArray_DESCR(ary)), &len);
#endif
        if(ds == NULL) {
            this->_status.add(daal::services::UnknownError);
            return daal::services::Status();
        }
        archive->set(len);
        archive->set(ds, len);
        // now the array data
        archive->set(PyArray_DIMS(ary)[0]);
        archive->set(PyArray_DIMS(ary)[1]);
        archive->set((char*)PyArray_DATA(ary), PyArray_DIMS(ary)[0] * PyArray_DIMS(ary)[1]);

        return daal::services::Status();
    }

    /** \private */
    daal::services::Status deserializeImpl(const daal::data_management::OutputDataArchive *archive)
    {
        // First deserialize the type descriptor in string representation...
        size_t len;
        archive->set(len);

        char * nds = new char[len];
        archive->set(nds, len);
        // ..then create the type descriptor
        PyObject * npy = PyImport_ImportModule("numpy");
        PyObject * globalDictionary = PyModule_GetDict(npy);
        PyArray_Descr* nd = (PyArray_Descr*)PyRun_String(PyString_AsString(PyObject_Str(PyString_FromString(nds))), Py_eval_input, globalDictionary,
                                                         NULL);
        delete [] nds;
        if(nd == NULL) {
            this->_status.add(daal::services::UnknownError);
            return daal::services::Status();
        }
        // now get the array data
        npy_intp dims[2];
        archive->set(dims[0]);
        archive->set(dims[1]);
        // create the array...
        _ary = (PyArrayObject*)PyArray_SimpleNewFromDescr(1, dims, nd);
        if(_ary == NULL) {
            this->_status.add(daal::services::UnknownError);
            return daal::services::Status();
        }
        // ...then copy data
        archive->set((char*)PyArray_DATA(_ary), dims[0]*dims[1]);

        return daal::services::Status();
    }

private:
    // This is a generic copy function for copying between DAAL and numpy
    // Wet template parameter WBack to true for copying back to numpy array.
    //
    // 1. Retrieve requested slide from numpy array by using python's C-API
    // 2. Create numpy array iterator setup for casting to requested type
    // 3. Iterate through numpy array and copy to/from block using memcpy
    template<typename T, bool WBack>
    void do_cpy(daal::data_management::BlockDescriptor<T>& block, size_t startcol, size_t ncols, size_t startrow, size_t nrows)
    {
        // Handle zero-sized arrays specially
        if (PyArray_SIZE(_ary) == 0) {
            return;
        }

        auto __state = PyGILState_Ensure();


        // Getting the slice/block from the numpy array requires creating slices
        // so it's not particularly cheap
        // Even though surprisingly complicated this is much simpler than
        // extracting the block manually. The numpy iterator doesn't seem
        // to allow specifying the slice as the IndexRange. We'd need to
        // cut by rows and then manually detect the columns in the inner loop.
        // Even if done this way, it's not clear which one is faster.
        // If performance becomes a problem, we might consider using cython instead.
        PyObject* s1s = PyLong_FromLong(startrow);
        PyObject* s1e = PyLong_FromLong(startrow+nrows);
        PyObject* s2s = PyLong_FromLong(startcol);
        PyObject* s2e = PyLong_FromLong(startcol+ncols);
        PyObject* slice = PyTuple_New(2);
        PyTuple_SET_ITEM(slice, 0, PySlice_New(s1s, s1e, NULL));
        PyTuple_SET_ITEM(slice, 1, PySlice_New(s2s, s2e, NULL));
        PyArrayObject * ary_block = (PyArrayObject*)PyObject_GetItem((PyObject*)_ary, slice);
        Py_XDECREF(s1s);
        Py_XDECREF(s1e);
        Py_XDECREF(s2s);
        Py_XDECREF(s2e);

        // create the iterator
        PyObject *val = Py_BuildValue("s", npy_type<T>::value);
        PyArray_Descr *dtype;
        PyArray_DescrConverter(val, &dtype);
        Py_XDECREF(val);

        NpyIter * iter = NpyIter_New(ary_block,
                                     ((WBack ? NPY_ITER_WRITEONLY : NPY_ITER_READONLY) // the array is never written to
                                      | NPY_ITER_EXTERNAL_LOOP  // Inner loop is done outside the iterator for efficiency.
                                      | NPY_ITER_RANGED         // Read a sub-range
                                      | NPY_ITER_BUFFERED),     // Buffer, don't copy
                                     NPY_CORDER,                // Visit elements in C memory order
                                     NPY_UNSAFE_CASTING,        // all casting allowed
                                     dtype);                    // let's numpy do the casting

        if (iter == NULL) {
            PyGILState_Release(__state);
            return;
        }

        // The iternext function gets stored in a local variable
        // so it can be called repeatedly in an efficient manner.
        NpyIter_IterNextFunc * iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            PyGILState_Release(__state);
            return;
        }
        // The location of the data pointer which the iterator may update
        char ** dataptr = NpyIter_GetDataPtrArray(iter);
        // The location of the stride which the iterator may update
        npy_intp * strideptr = NpyIter_GetInnerStrideArray(iter);
        // The location of the inner loop size which the iterator may update
        npy_intp * innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

        if(NpyIter_GetDescrArray(iter)[0]->elsize != sizeof(T)) {
            NpyIter_Deallocate(iter);
            throw std::invalid_argument("Encountered unexpected element size or type when copying block.");
            PyGILState_Release(__state);
            return;
        }

        PyGILState_Release(__state);

        // ptr to column in block
        T * blockPtr = block.getBlockPtr();

        // we assume all inner strides are identical
        npy_intp innerstride = strideptr[0];
        if(strideptr[0] == sizeof(T)) {
            do {
                npy_intp size = *innersizeptr;
                memcpy(WBack ? *dataptr        : (char*)blockPtr,
                       WBack ? (char*)blockPtr : *dataptr,
                       sizeof(T) * size);
                blockPtr += size;
            } while(iternext(iter));
        } else {
            do {
                // For efficiency, should specialize this based on item size...
                npy_intp i;
                char *src = *dataptr;
                npy_intp size = *innersizeptr;
                for(i = 0; i < size; ++i, src += innerstride, blockPtr += 1) {
                    memcpy(WBack ? src             : (char*)blockPtr,
                           WBack ? (char*)blockPtr : src,
                           sizeof(T));
                }
            } while(iternext(iter));
        }

        __state = PyGILState_Ensure();
        NpyIter_Deallocate(iter);
        PyGILState_Release(__state);
        return;
    }


    template <typename T>
    daal::services::Status getTBlock(size_t idx, size_t numrows, int rwFlag, daal::data_management::BlockDescriptor<T>& block, size_t firstcol=0, size_t numcols=0xffffffff)
    {
        // sanitize bounds
        const size_t ncols = firstcol + numcols <= getNumberOfColumns() ? numcols : getNumberOfColumns() - firstcol;
        const size_t nrows = idx + numrows <= getNumberOfRows()         ? numrows : getNumberOfRows() - idx;

        // set shape of blockdescr
        block.setDetails(firstcol, idx, rwFlag);

        if(idx >= getNumberOfRows() || firstcol >= getNumberOfColumns()) {
            block.resizeBuffer( ncols, 0 );
            return daal::services::Status();
        }

        if(!block.resizeBuffer(ncols, nrows)) {
            return daal::services::Status(daal::services::ErrorMemoryAllocationFailed);
        }

        if(!(rwFlag & (int)daal::data_management::readOnly)) return daal::services::Status();

        // use our copy method in copy-out mode
        do_cpy<T, false>(block, firstcol, ncols, idx, nrows);
        return daal::services::Status();
    }


    template <typename T>
    daal::services::Status releaseTBlock(daal::data_management::BlockDescriptor<T>& block)
    {
        if(block.getRWFlag() & (int)daal::data_management::writeOnly) {
            const size_t ncols = block.getNumberOfColumns();
            const size_t nrows = block.getNumberOfRows();

            // use our copy method in write-back mode
            do_cpy<T, true>(block, block.getColumnsOffset(), ncols, 0, nrows);

            block.reset();
        }
        return daal::services::Status();
    }
};

#endif // __NPY4DAAL_H_INCLUDED__
