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

#ifndef _HLAPI_H_INCLUDED_
#define _HLAPI_H_INCLUDED_

#ifdef _WIN32
#define NOMINMAX
#endif

#include <daal.h>
using daal::step1Local;
using daal::step2Local;
using daal::step3Local;
using daal::step4Local;
using daal::step2Master;
using daal::step3Master;
using daal::services::LibraryVersionInfo;
#include "daal_compat.h"

#include <iostream>
#include <stdexcept>
#include <vector>
#include <limits>
#include <string>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define NTYPE PyObject*

#if PY_VERSION_HEX < 0x03000000
#define PyUnicode_Check(_x) PyString_Check(_x)
#define PyUnicode_AsUTF8(_x) PyString_AsString(_x)
#define PyUnicode_FromString(_x) PyString_FromString(_x)
#endif

extern "C" {
void c_daalinit(bool spmd=false, int flag=0, int nthreads=-1);
void c_daalfini();
size_t c_num_procs();
size_t c_my_procid();
}

typedef daal::services::SharedPtr< std::vector< std::vector< daal::byte > > > BytesArray;

template< typename T >
bool use_default(const daal::services::SharedPtr<T> * attr)
{
    return attr == NULL || attr->get() == NULL;
}

template< typename T >
bool use_default(const daal::services::SharedPtr<T> & attr)
{
    return attr.get() == NULL;
}

template< typename T >
bool use_default(const T * attr)
{
    return attr == NULL;
}

inline bool use_default(const std::string & attr)
{
    return attr.length() == 0;
}

inline bool use_default(const int & attr)
{
    return attr == -1;
}

inline bool use_default(const size_t & attr)
{
    return (long)attr == (long)-1;
}

#ifndef _WIN32
inline bool use_default(const DAAL_UINT64 & attr)
{
    return (long)attr == (long)-1;
}
#endif

inline bool use_default(const double & attr)
{
    return attr != attr;
}

inline bool use_default(const float & attr)
{
    return attr != attr;
}

inline bool string2bool(const std::string & s)
{
    if(s == "True" || s == "true" || s == "1") return true;
    if(s == "False" || s == "false" || s == "0") return false;
    throw std::invalid_argument("Bool must be one of {'True', 'true', '1', 'False', 'false', '0'}");
}

class algo_manager__iface__
{
public:
    inline algo_manager__iface__() {}
    inline virtual ~algo_manager__iface__() {}
    // We don't want any manager to be copied
    algo_manager__iface__(const algo_manager__iface__ &) = delete;
    void operator=(const algo_manager__iface__ &) = delete;
};

#if 0
static inline NTYPE as_native_shared_ptr(services::SharedPtr< const algo_manager__iface__ > algo)
{
    int gc = 0;
    MK_DAALPTR(ret, new services::SharedPtr< const algo_manager__iface__ >(algo), services::SharedPtr< algo_manager__iface__ >, gc);
    TMGC(gc);
    return ret;
}
#endif

// Our Batch input/Output manager, abstracts from input/output types
// also defines how to get results and finalize
template< typename A, typename I, typename O >
struct IOManager
{
    typedef O result_type;
    typedef I input1_type;
    typedef input1_type input_type;

    static result_type getResult(A & algo)
    {
        return daal::services::staticPointerCast<typename result_type::ElementType>(algo.getResult());
    }
    static bool needsFini()
    {
        return true;
    }
};

struct TableOrFList
{
    mutable daal::data_management::NumericTablePtr table;
    std::string                            file;
    std::vector< daal::data_management::NumericTablePtr > tlist;
    std::vector< std::string >             flist;
    inline TableOrFList(daal::data_management::NumericTablePtr t)
        : table(t), file(), tlist(), flist() {}
    inline TableOrFList()
        : table(), file(), tlist(), flist() {}
    TableOrFList(PyObject *);
};

// return input as DAAL numeric table.
extern const daal::data_management::NumericTablePtr get_table(const TableOrFList & t);

template< typename T >
struct RAW
{
    typedef T TYPE;
    const TYPE operator()(const T & o) {return o;}
};

template< typename T >
struct RAW< daal::services::SharedPtr< T > >
{
    typedef daal::services::SharedPtr< T > * TYPE;
    TYPE operator()(daal::services::SharedPtr< T > o) {return new daal::services::SharedPtr< T >(o);}
};

template< typename T > T to_daal(T t) {return t;}
template< typename T > daal::services::SharedPtr<T> to_daal(daal::services::SharedPtr<T>* t) {return *t;}
inline const TableOrFList & to_daal(const TableOrFList * t) {return *t;}
inline const TableOrFList & to_daal(TableOrFList * t) {return *t;}

template< typename T >
void * get_nt_data_ptr(const daal::data_management::NumericTablePtr * ptr)
{
    auto dptr = dynamic_cast< const daal::data_management::HomogenNumericTable< T >* >((*ptr).get());
    return dptr ? reinterpret_cast< void* >(dptr->getArraySharedPtr().get()) : NULL;
}

extern int64_t string2enum(const std::string& str, std::map< std::string, int64_t > & strmap);

static std::string to_std_string(PyObject * o)
{
    return PyUnicode_AsUTF8(o);
}

const double NaN64 = std::numeric_limits<double>::quiet_NaN();
const float NaN32 = std::numeric_limits<float>::quiet_NaN();

typedef daal::data_management::DataCollectionPtr data_management_DataCollectionPtr;
typedef daal::data_management::NumericTablePtr data_management_NumericTablePtr;
typedef daal::data_management::KeyValueDataCollectionPtr dict_NumericTablePtr;

extern "C" void to_c_array(const daal::data_management::NumericTablePtr * ptr, void ** data, size_t * dims, char dtype = 0);
extern PyObject * make_nda(daal::data_management::NumericTablePtr * nt_ptr);
extern PyObject * make_nda(daal::data_management::KeyValueDataCollectionPtr * nt_ptr, const std::map< int64_t, std::string > &);
extern daal::data_management::NumericTablePtr * make_nt(PyObject * nda);
extern daal::data_management::KeyValueDataCollectionPtr * make_dnt(PyObject * dict, std::map< std::string, int64_t > &);

extern const daal::data_management::NumericTablePtr readCSV(const std::string& fname);


template<class T, class U>
T* dynamicPointerPtrCast(U *r)
{
    T tmp = daal::services::dynamicPointerCast<typename T::ElementType>(*r);
    return tmp ? new T(*reinterpret_cast<T*>(r)) : NULL;
}
#endif // _HLAPI_H_INCLUDED_
