#ifndef DAAL4PY_PICKLING_SUPPORT_INC_
#define DAAL4PY_PICKLING_SUPPORT_INC_

#include "daal.h"
#include "Python.h"

template <typename T>
PyObject *
serialize_si(daal::services::SharedPtr< T > * ptr) 
{
    daal::data_management::InputDataArchive dataArch;

    (daal::services::staticPointerCast<daal::data_management::SerializationIface>(*ptr))->serialize(dataArch);

    Py_ssize_t buf_len = dataArch.getSizeOfArchive();
    daal::services::SharedPtr<byte> shr_ptr = dataArch.getArchiveAsArraySharedPtr();

    return PyBytes_FromStringAndSize(reinterpret_cast<char *>(shr_ptr.get()), buf_len);
}

template <typename T>
T *
deserialize_si(PyObject * py_bytes)
{
    char* buf;
    Py_ssize_t buf_len;
    PyBytes_AsStringAndSize(py_bytes, &buf, &buf_len);

    daal::data_management::OutputDataArchive dataArch(reinterpret_cast<daal::byte *>(buf), buf_len);

    return new T(daal::services::staticPointerCast<typename T::ElementType >(dataArch.getAsSharedPtr()));
}

#endif
