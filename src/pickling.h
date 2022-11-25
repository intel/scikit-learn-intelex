/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#ifndef DAAL4PY_PICKLING_INC_
#define DAAL4PY_PICKLING_INC_

#include "daal.h"
#include "Python.h"

template <typename T>
PyObject * serialize_si(daal::services::SharedPtr<T> * ptr)
{
    if (!ptr || !(*ptr)) return NULL;

    daal::data_management::InputDataArchive dataArch;

    (daal::services::staticPointerCast<daal::data_management::SerializationIface>(*ptr))->serialize(dataArch);

    Py_ssize_t buf_len                            = dataArch.getSizeOfArchive();
    daal::services::SharedPtr<daal::byte> shr_ptr = dataArch.getArchiveAsArraySharedPtr();

    return PyBytes_FromStringAndSize(reinterpret_cast<char *>(shr_ptr.get()), buf_len);
}

template <typename T>
T * deserialize_si(PyObject * py_bytes)
{
    if (!py_bytes || py_bytes == Py_None) return NULL;

    char * buf;
    Py_ssize_t buf_len;
    PyBytes_AsStringAndSize(py_bytes, &buf, &buf_len);

    daal::data_management::OutputDataArchive dataArch(reinterpret_cast<daal::byte *>(buf), buf_len);

    return new T(daal::services::staticPointerCast<typename T::ElementType>(dataArch.getAsSharedPtr()));
}

#endif // DAAL4PY_PICKLING_INC_
