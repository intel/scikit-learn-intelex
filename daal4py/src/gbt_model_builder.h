/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef _GBT_MODEL_BUILDER_INCLUDED_
#define _GBT_MODEL_BUILDER_INCLUDED_

#include <daal.h>

typedef daal::algorithms::gbt::classification::ModelBuilder c_gbt_classification_model_builder;
typedef daal::algorithms::gbt::regression::ModelBuilder c_gbt_regression_model_builder;

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

#endif // _GBT_MODEL_BUILDER_INCLUDED_
