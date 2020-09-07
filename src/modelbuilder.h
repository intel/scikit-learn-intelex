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

typedef daal::algorithms::gbt::classification::ModelBuilder c_gbt_classification_ModelBuilder;
typedef daal::algorithms::gbt::regression::ModelBuilder c_gbt_regression_ModelBuilder;

typedef c_gbt_classification_ModelBuilder::NodeId c_gbt_clf_NodeId;
typedef c_gbt_classification_ModelBuilder::TreeId c_gbt_clf_TreeId;
typedef c_gbt_regression_ModelBuilder::NodeId c_gbt_reg_NodeId;
typedef c_gbt_regression_ModelBuilder::TreeId c_gbt_reg_TreeId;

#define c_gbt_clf_noParent c_gbt_classification_ModelBuilder::noParent
#define c_gbt_reg_noParent c_gbt_regression_ModelBuilder::noParent

static daal::algorithms::gbt::classification::ModelPtr * get_gbt_classification_modelbuilder_Model(daal::algorithms::gbt::classification::ModelBuilder * obj_)
{
    return RAW<daal::algorithms::gbt::classification::ModelPtr>()(obj_->getModel());
}

static daal::algorithms::gbt::regression::ModelPtr * get_gbt_regression_modelbuilder_Model(daal::algorithms::gbt::regression::ModelBuilder * obj_)
{
    return RAW<daal::algorithms::gbt::regression::ModelPtr>()(obj_->getModel());
}

#endif // _MODELBUILDER_INCLUDED_
