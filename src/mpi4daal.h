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

#ifndef _MPI4DAAL_INCLUDED_
#define _MPI4DAAL_INCLUDED_

#include "mpi.h"

struct MPI
{
    template<typename T>
    static std::vector<T> gather(int rank, int nRanks, const T& p_res )
    {
        // Serialize the partial result into a data archive
        daal::data_management::InputDataArchive in_arch;
        p_res->serialize(in_arch);

        // gather all partial results
        // First get all sizes, then gather on root
        int sizes[nRanks];
        int offsets[nRanks];
        int mysize = in_arch.getSizeOfArchive();
        MPI_Gather(&mysize, 1, MPI_INT, sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int tot_sz = 0;
        char * buff = NULL;

        if(rank == 0) {
            for(int i = 0; i < nRanks; i++) {
                offsets[i] = i > 0 ? sizes[i-1] : 0;
                tot_sz += sizes[i];
            }
            buff = new char[tot_sz];
        }

        MPI_Gatherv(in_arch.getArchiveAsArraySharedPtr().get(), mysize, MPI_CHAR,
                    buff, sizes, offsets, MPI_CHAR,
                    0, MPI_COMM_WORLD);

        std::vector<T> all;
        if(rank == 0) {
            all.resize(nRanks);
            for(int i=0; i<nRanks; ++i) {
                // FIXME: This is super-inefficient, we need to write our own DatArchive to avoid extra copy
                daal::data_management::OutputDataArchive out_arch(reinterpret_cast<daal::byte*>(buff+offsets[i]), sizes[i]);
                all[i] = daal::services::dynamicPointerCast<typename T::ElementType>(out_arch.getAsSharedPtr());
            }
            delete [] buff;
        }

        return all;
    }

    template<typename T>
    static T bcast(int rank, int nRanks, T obj)
    {
        if(rank == 0) {
            // Serialize the partial result into a data archive
            daal::data_management::InputDataArchive in_arch;
            obj->serialize(in_arch);
            int size = in_arch.getSizeOfArchive();
            MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(in_arch.getArchiveAsArraySharedPtr().get(), size, MPI_CHAR, 0, MPI_COMM_WORLD);
        } else {
            int size = 0;
            MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            char * buff = new char[size];
            MPI_Bcast(buff, size, MPI_CHAR, 0, MPI_COMM_WORLD);
            daal::data_management::OutputDataArchive out_arch(reinterpret_cast<daal::byte*>(buff), size);
            obj = daal::services::dynamicPointerCast<typename T::ElementType>(out_arch.getAsSharedPtr());
        }
        return obj;
    }

    template<typename T> struct std2mpi;
    template<>struct stdmpi<double> { const MPI_Datatype typ = MPI_DOUBLE; }
    template<>struct stdmpi<float> { const MPI_Datatype typ = MPI_FLOAT; }
    template<>struct stdmpi<int> { const MPI_Datatype typ = MPI_INT; }

    template<typename T>
    void allreduce(T * buf, size_t n, MPI_Op op)
    {
        MPI_Allreduce(MPI_IN_PLACE, buf, (int)n, stdmpi<T>::typ, MPI_COMM_WORLD);
    }
};

#endif // _MPI4DAAL_INCLUDED_
