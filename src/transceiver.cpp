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

#include "mpi/mpi_transceiver.h"
#include <tbb/mutex.h>

static transceiver * s_trsc = NULL;

transceiver * get_transceiver()
{
    static tbb::mutex mtx;

    if(s_trsc == NULL) {
        tbb::mutex::scoped_lock lock(mtx);
        if(s_trsc == NULL) {
            s_trsc = new transceiver(std::make_shared<mpi_transceiver>());
        }
    }
    return s_trsc;
}

void del_transceiver()
{
    delete s_trsc;
}
// FIXME: GC: auto delete transceiver
