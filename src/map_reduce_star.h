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

#ifndef _MAP_REDUCE_STAR_INCLUDED_
#define _MAP_REDUCE_STAR_INCLUDED_

#include "mpi4daal.h"

namespace map_reduce_star {

template<typename Algo>
class map_reduce_star
{
public:
    typename Algo::iomstep2Master_type::result_type
    static compute(Algo & algo, TableOrFList * data, TableOrFList * labels)
    {
        int rank, nRanks;
        MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        auto s1_result = algo.run_step1Local(inputs...);
        // gather all partial results
        auto p_results = MPI::gather(rank, nRanks, s1_result);
        // call reduction on root
        typename Algo::iomstep2Master_type::result_type res;
        if(rank == 0) res = algo.run_step2Master(p_results);
        // bcast final result
        return MPI::bcast(rank, nRanks, res);
    }

    template<typename ... Ts>
    static typename Algo::iomstep2Master_type::result_type
    compute(Algo & algo, Ts& ... inputs)
    {
        return map_reduce(algo, get_table(inputs)...);
    }
};

} // namespace map_reduce_star {

#endif // _MAP_REDUCE_STAR_INCLUDED_
