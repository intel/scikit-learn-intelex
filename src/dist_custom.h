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

#ifndef _DIST_CUSTOM_INCLUDED_
#define _DIST_CUSTOM_INCLUDED_

#include "mpi4daal.h"
#include <algorithm>

namespace dist_custom {

template<typename Algo>
class dist_custom;

// custom distribution class for logistic regression
template<>
template< typename fptype, daal::algorithms::logistic_regression::training::Method method >
dist_custom< logistic_regression_training_manager< fptype, method > >
{
public:
    typedef logistic_regression_training_manager< fptype, method > Algo;

    // the "map" phase: computing loss/gradient
    static NumericTablePtr compute_loss(Algo & algo,
                                        daal::data_management::NumericTablePtr & x,
                                        daal::data_management::NumericTablePtr & y,
                                        daal::data_management::NumericTablePtr & weights)
    {
        // we use logistic_loss directly
        auto loss = daal::algorithms::optimization_solver::logistic_loss::Batch<>::create(x->getNumberOfRows());
        // set parameters as in our logistic regression class
        loss->parameter().interceptFlag = algo._interceptFlag;
        loss->parameter().penaltyL1 = algo._penaltyL1;
        loss->parameter().penaltyL2 = algo._penaltyL2;
        loss->parameter().resultsToCompute = daal::algorithms::optimization_solver::objective_function::value
                                             | daal::algorithms::optimization_solver::objective_function::gradient;
        loss->input.set(logistic_loss::data, x);
        loss->input.set(logistic_loss::dependentVariables, y);
        loss->input.set(logistic_loss::argument, weights);

        loss->compute();

        return loss->getResult()->get(objective_function::gradientIdx);
    };

    
     typename Algo::iomb_type::result_type map_reduce(Algo & algo,
                                                      daal::data_management::NumericTablePtr & x,
                                                      daal::data_management::NumericTablePtr & y,
                                                      size_t epoch_number,
                                                      float learning_rate)
    {
        int rank, nRanks;
        MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        /* In case of logistic_loss and interceptFlag == true,
         * the number of weights is nFeatures + 1 */
        // FIXME: else?
        size_t N = x.getNumberOfColumns() + 1;
        double weights = new double[N];
        std::fill_n(weights, x.getNumberOfColumns() + 1, 1e-2);

        for (size_t e = 0; e < epoch_number; e++) {
            for (size_t b = 0; b < 1; b++) { // FIXME iterate over batches
                const auto gradient_tbl = compute_loss(algo, x, y, weights);
                double * gradient = dynamicPointerCast< daal::data_management::HomogenNumericTable<double> >(gradient_tbl)->getArray();
                MPI::allreduce(gradient, N, MPI_SUM); // in-place
                for (size_t i = 0; i < N; i++) { // FIXME: #weights == #gradient?
                    weights[i] += learning_rate * (gradient[i] / nRanks); // div by nRanks because we need avrg
                }
            }
        }

        // FIXME: How to create a daal::algorithms::logistic_regression::Model??????
        // FIXME: HOw to create a daal::algorithms::logistic_regression::training::Result??
    }

    template<typename ... Ts>
    static typename Algo::iomb_type::result_type
    compute(Algo & algo, const TableOrFList & y, const TableOrFList & y)
    {
        return map_reduce(algo, get_table(x), get_table(y));
    }
};

} // namespace dist_custom {

#endif // _DIST_CUSTOM_INCLUDED_
