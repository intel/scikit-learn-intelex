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

template<typename T> struct std2mpi;
template<>struct std2mpi<double> { static const MPI_Datatype typ = MPI_DOUBLE; };
template<>struct std2mpi<float>  { static const MPI_Datatype typ = MPI_FLOAT; };
template<>struct std2mpi<int>    { static const MPI_Datatype typ = MPI_INT; };
template<>struct std2mpi<bool>   { static const MPI_Datatype typ = MPI_C_BOOL; };
template<>struct std2mpi<size_t> { static const MPI_Datatype typ = MPI_UNSIGNED_LONG; };

template<typename T> bool not_empty(const daal::services::SharedPtr<T> & obj) { return obj; };
template<typename T> bool not_empty(const daal::data_management::interface1::NumericTablePtr & obj) { return obj && obj->getNumberOfRows() && obj->getNumberOfColumns(); };


struct MPI4DAAL
{
    static void init()
    {
        int is_initialized;
        MPI_Initialized(&is_initialized);
        if(!is_initialized) MPI_Init(NULL, NULL);
    }

    static void fini()
    {
        MPI_Finalize();
    }

    static int nRanks()
    {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        return size;
    }

    static size_t rank()
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return rank;
    }

    template<typename T>
    static void send(const T& obj, int recpnt, int tag)
    {
        daal::data_management::InputDataArchive in_arch;
        int mysize(0);
        // Serialize the DAAL object into a data archive
        if(not_empty(obj)) {
            obj->serialize(in_arch);
            mysize = in_arch.getSizeOfArchive();
        }
        // and send it away to our recipient
        MPI_Send(&mysize, 1, MPI_INT, recpnt, tag, MPI_COMM_WORLD);
        if(mysize) {
            MPI_Send(in_arch.getArchiveAsArraySharedPtr().get(), mysize, MPI_CHAR, recpnt, tag, MPI_COMM_WORLD);
        }
    }

    template<typename T>
    static T recv(int sender, int tag)
    {
        int sz(0);
        MPI_Recv(&sz, 1, MPI_INT, sender, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        T res;
        if(sz) {
            daal::byte * buf = new daal::byte[sz];
            MPI_Recv(buf, sz, MPI_CHAR, sender, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            daal::data_management::OutputDataArchive out_arch(buf, sz);
            res = daal::services::dynamicPointerCast<typename T::ElementType>(out_arch.getAsSharedPtr());
            delete [] buf;
        }
        return res;
    }

    template<typename T>
    static std::vector<T> gather(int rank, int nRanks, const T& p_res )
    {
        // Serialize the partial result into a data archive
        daal::data_management::InputDataArchive in_arch;
        p_res->serialize(in_arch);

        // gather all partial results
        // First get all sizes, then gather on root
        int * sizes = new int[nRanks];
        int * offsets = new int[nRanks];
        int mysize = in_arch.getSizeOfArchive();
        MPI_Gather(&mysize, 1, MPI_INT, sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int tot_sz = mysize;
        char * buff = NULL;

        if(rank == 0) {
            offsets[0] = 0;
            for(int i = 1; i < nRanks; ++i) {
                offsets[i] = offsets[i-1] + sizes[i-1];
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

		delete [] sizes;
		delete [] offsets;

        return all;
    }


    template<typename T>
    static T bcast(int rank, int nRanks, const T & obj, int root=0)
    {
        T out = obj;
        MPI_Bcast(&out, 1, std2mpi<T>::typ, root, MPI_COMM_WORLD);
        return out;
    }

    template<typename T>
    static daal::services::SharedPtr<T> bcast(int rank, int nRanks, daal::services::SharedPtr<T> obj, int root=0)
    {
        if(rank == root) {
            // Serialize the partial result into a data archive
            daal::data_management::InputDataArchive in_arch;
            obj->serialize(in_arch);
            int size = in_arch.getSizeOfArchive();
            MPI_Bcast(&size, 1, MPI_INT, root, MPI_COMM_WORLD);
            if(size > 0) MPI_Bcast(in_arch.getArchiveAsArraySharedPtr().get(), size, MPI_CHAR, root, MPI_COMM_WORLD);
        } else {
            int size = 0;
            MPI_Bcast(&size, 1, MPI_INT, root, MPI_COMM_WORLD);
            if(size) {
                char * buff = new char[size];
                MPI_Bcast(buff, size, MPI_CHAR, root, MPI_COMM_WORLD);
                daal::data_management::OutputDataArchive out_arch(reinterpret_cast<daal::byte*>(buff), size);
                obj = daal::services::dynamicPointerCast<T>(out_arch.getAsSharedPtr());
            } else {
                obj.reset();
            }
        }
        return obj;
    }

    template<typename T>
    static void allreduce(T * buf, size_t n, MPI_Op op)
    {
        MPI_Allreduce(MPI_IN_PLACE, buf, (int)n, std2mpi<T>::typ, op, MPI_COMM_WORLD);
    }

    template<typename T>
    static void exscan(T * buf, size_t n, MPI_Op op)
    {
        MPI_Exscan(MPI_IN_PLACE, buf, n, std2mpi<T>::typ, op, MPI_COMM_WORLD);
    }

#if 0 // untested
    template<typename T>
    static T scatter(int rank, int nRanks, const std::vector<T> & objs )
    {
        const int STAG = 7007;
        T res;
        if(rank == 0) {
            size_t n = objs.size();
            MPI_Request reqs = new MPI_Request[n*2];
            for(size_t i = 1; i < n; ++i) {
                // Serialize each obj into a data archive
                daal::data_management::InputDataArchive in_arch;
                int mysize(0);
                if(not_empty(objs[i])) {
                    objs[i]->serialize(in_arch);
                    mysize = in_arch.getSizeOfArchive();
                }
                MPI_Isend(&mysize, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &reqs[i]);
                if(mysize) {
                    // we send payload only if size>0
                    MPI_Isend(in_arch.getArchiveAsArraySharedPtr().get(), mysize, MPI_CHAR, i, STAG, MPI_COMM_WORLD, &reqs[n+i]);
                } else {
                    reqs[n+i] = MPI_REQUEST_NULL;
                }
            }
            // root returns its own object as-is
            res = objs[0];
            reqs[0] = req[n] = MPI_REQUEST_NULL;
            MPI_Waitall(n*2, reqs, MPI_STATUSES_IGNORE);
        } else {
            res = recv<T>(0, STAG);
        }

        return res;
    }
#endif

};

#endif // _MPI4DAAL_INCLUDED_
