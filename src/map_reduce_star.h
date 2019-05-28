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

#ifndef _MAP_REDUCE_STAR_INCLUDED_
#define _MAP_REDUCE_STAR_INCLUDED_

#include "transceiver.h"

namespace map_reduce_star {

template<typename Algo>
class map_reduce_star
{
public:
    template<typename ... Ts>
    typename Algo::iomstep2Master_type::result_type
    static map_reduce(Algo & algo, Ts& ... inputs)
    {
        auto tcvr = get_transceiver();
        auto s1_result = algo.run_step1Local(inputs...);
        // gather all partial results
        auto p_results = tcvr->gather(s1_result);
        // call reduction on root
        typename Algo::iomstep2Master_type::result_type res;
        if(tcvr->me() == 0) res = algo.run_step2Master(p_results);
        // bcast final result
        tcvr->bcast(res);
        return res;
    }

    template<typename ... Ts>
    static typename Algo::iomstep2Master_type::result_type
    compute(Algo & algo, Ts& ... inputs)
    {
        return map_reduce(algo, get_table(inputs)...);
    }
};

} // namespace map_reduce_star {

namespace map_reduce_star_plus {

    template<typename Algo>
    class map_reduce_star_plus
    {
    public:
        template<typename ... Ts>
        typename Algo::iomstep3Local_type::result_type
        static map_reduce(Algo & algo, Ts& ... inputs)
        {
            int rank = MPI4DAAL::rank();
            int nRanks = MPI4DAAL::nRanks();

            // run step1 and gather all partial results
            auto s1Res = algo.run_step1Local(inputs...);
            // we need to replace in some way exact daal::algorithms::svd::outputOfStep1ForStep2 by common definition
            auto s1OutForStep2 = s1Res->get(daal::algorithms::svd::outputOfStep1ForStep2);
            auto s2InFromStep1 = MPI4DAAL::gather(rank, nRanks, s1OutForStep2);

            typename Algo::iomstep2Master_type::result_type res;
            const int S23TAG = 4004; //what it should be? unique?
            daal::data_management::DataCollectionPtr inputOfStep3FromStep2;
            if(rank == 0) {
                res = algo.run_step2Master(s2InFromStep1);
                // get intputs for step3 and send them to all processes
                auto outputOfStep2ForStep3 = std::get<1>(res)->get(daal::algorithms::svd::outputOfStep2ForStep3);
                inputOfStep3FromStep2 = daal::services::staticPointerCast<daal::data_management::DataCollection>((*outputOfStep2ForStep3)[0]);
                for(size_t i = 1; i < nRanks; i++) {
                    MPI4DAAL::send((*outputOfStep2ForStep3)[i], i, S23TAG);
                }
            } else {
                inputOfStep3FromStep2 = MPI4DAAL::recv<daal::data_management::DataCollectionPtr>(0, S23TAG);
            }

            // run step3
            auto inputOfStep3FromStep1 = s1Res->get(daal::algorithms::svd::outputOfStep1ForStep3);
            auto step3Output = algo.run_step3Local(inputOfStep3FromStep1, inputOfStep3FromStep2);

            // we need to return std::get<0>(res) (rightSingularMatrix and singularValues) and step3Output (leftSingularMatrix) at the same time
            // and them must be available in Python...
            return step3Output;
        }

        template<typename ... Ts>
        static typename Algo::iomstep3Local_type::result_type
        compute(Algo & algo, Ts& ... inputs)
        {
            MPI4DAAL::init();
            return map_reduce(algo, get_table(inputs)...);
        }
    };

} // namespace map_reduce_star_plus {

#endif // _MAP_REDUCE_STAR_INCLUDED_
