/*******************************************************************************
* Copyright 2020 Intel Corporation
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

// Definitions/declarations, mapping cython names/types to DAAL's actual types

#ifndef _MODELBUILDER_INCLUDED_
#define _MODELBUILDER_INCLUDED_

#include <daal.h>
#include <Python.h>
#include "daal4py.h"

typedef daal::algorithms::gbt::classification::ModelBuilder c_gbt_classification_model_builder;
typedef daal::algorithms::gbt::regression::ModelBuilder c_gbt_regression_model_builder;
typedef daal::algorithms::logistic_regression::ModelBuilder<DAAL_ALGORITHM_FP_TYPE> c_logistic_regression_model_builder;

typedef c_gbt_classification_model_builder::NodeId c_gbt_clf_node_id;
typedef c_gbt_classification_model_builder::TreeId c_gbt_clf_tree_id;
typedef c_gbt_regression_model_builder::NodeId c_gbt_reg_node_id;
typedef c_gbt_regression_model_builder::TreeId c_gbt_reg_tree_id;

#define c_gbt_clf_no_parent c_gbt_classification_model_builder::noParent
#define c_gbt_reg_no_parent c_gbt_regression_model_builder::noParent

static daal::algorithms::gbt::classification::ModelPtr * get_gbt_classification_model_builder_model(daal::algorithms::gbt::classification::ModelBuilder * obj_)
{
    return RAW<daal::algorithms::gbt::classification::ModelPtr>()(obj_->getModel());
}

static daal::algorithms::gbt::regression::ModelPtr * get_gbt_regression_model_builder_model(daal::algorithms::gbt::regression::ModelBuilder * obj_)
{
    return RAW<daal::algorithms::gbt::regression::ModelPtr>()(obj_->getModel());
}

template <typename modelFPType = DAAL_ALGORITHM_FP_TYPE>
static daal::algorithms::logistic_regression::ModelPtr * get_logistic_regression_model_builder_model(
    daal::algorithms::logistic_regression::ModelBuilder<modelFPType> * obj_)
{
    return RAW<daal::algorithms::logistic_regression::ModelPtr>()(obj_->getModel());
}

static daal::data_management::NumericTablePtr getTable(const data_or_file & t)
{
    return get_table(t);
}

#endif // _MODELBUILDER_INCLUDED_
