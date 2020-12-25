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

#ifndef _ONECCL_TRANSCEIVER_INCLUDED_
#define _ONECCL_TRANSCEIVER_INCLUDED_

#include "transceiver.h"

// implementation of transceiver_iface using oneccl
class oneccl_transceiver : public transceiver_impl
{
public:
    oneccl_transceiver() : transceiver_impl() {}

    virtual void init();

    virtual void fini();

    virtual size_t nMembers();
    
    virtual size_t me();

    virtual void bcast(void * ptr, size_t N, size_t root);

    virtual void send(const void* buff, size_t N, size_t recpnt, size_t tag);

    virtual size_t recv(void * buff, size_t N, int sender, int tag);

    virtual void * gather(const void * ptr, size_t N, size_t root, const size_t * sizes, bool varying=true);

    virtual void reduce_all(void * inout, type_type T, size_t N, operation_type op);

    virtual void reduce_exscan(void * inout, type_type T, size_t N, operation_type op);
};

#endif // _ONECCL_TRANSCEIVER_INCLUDED_

