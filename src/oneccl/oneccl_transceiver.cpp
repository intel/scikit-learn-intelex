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

#include "oneccl_transceiver.h"
#include "daal4py_defines.h"
#include <ccl.h>
#include <Python.h>
#include <limits>
#include <iostream>



void oneccl_transceiver::init()
{
    ccl_init();
    transceiver_impl::init();	
}

void oneccl_transceiver::fini()
{
    ccl_finalize();
}

size_t oneccl_transceiver::me()
{
    size_t me;
    ccl_get_comm_rank(NULL, &me);
    return me;
}

void oneccl_transceiver::bcast(void * ptr, size_t N, size_t root)
{
    ccl_request_t request;
    ccl_bcast(ptr, N, ccl_dtype_char, root, NULL, NULL, NULL, &request);
    ccl_wait(request);
}

size_t oneccl_transceiver::nMembers()
{
    size_t size;
    ccl_get_comm_size(NULL, &size);
    return size;
}

void oneccl_transceiver::send(const void* buff, size_t N, size_t recpnt, size_t tag)
{
    throw std::logic_error("transceiver_oneccl::send not yet implemented");
}

size_t oneccl_transceiver::recv(void * buff, size_t N, int sender, int tag)
{
    throw std::logic_error("transceiver_oneccl::recv not yet implemented");
    // return 0;
}

void * oneccl_transceiver::gather(const void * ptr, size_t N, size_t root, const size_t * sizes, bool varying)
{
    char * buff = NULL;

    buff = static_cast<char *>(daal::services::daal_malloc(m_nMembers*N));
    DAAL4PY_CHECK_MALLOC(buff);

    size_t* recvCounts = new size_t[m_nMembers];
    
    for (size_t i = 0; i < m_nMembers; i++)
    {
         recvCounts[i] = varying ? sizes[i] : N;
    }

    ccl_request_t request;
    ccl_allgatherv(ptr, N, buff, recvCounts, ccl_dtype_char, NULL, NULL, NULL, &request);
    ccl_wait(request);
    
    return buff;
}

void oneccl_transceiver::reduce_all(void * inout, type_type T, size_t N, operation_type op)
{
    throw std::logic_error("transceiver_oneccl::reduce_all not yet implemented");
}

void oneccl_transceiver::reduce_exscan(void * inout, type_type T, size_t N, operation_type op)
{
    throw std::logic_error("transceiver_oneccl::reduce_exscan not yet implemented");
}

// ************************************
// ************************************

extern "C" PyMODINIT_FUNC PyInit_oneccl_transceiver(void)
{
    // shared pointer, will GC transceiver when shutting down
    static std::shared_ptr<oneccl_transceiver> s_smt;
    PyObject *m;
    static struct PyModuleDef moduledef = { PyModuleDef_HEAD_INIT, "oneccl_transceiver", "No docs", -1, NULL, };
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    s_smt.reset(new oneccl_transceiver);
    PyObject_SetAttrString(m, "transceiver", PyLong_FromVoidPtr((void*)(&s_smt)));
    return m;
}

