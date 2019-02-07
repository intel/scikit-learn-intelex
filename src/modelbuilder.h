#include <daal.h>

typedef daal::algorithms::decision_forest::classification::ModelBuilder c_decision_forest_classification_ModelBuilder;
typedef c_decision_forest_classification_ModelBuilder::NodeId c_NodeId;
typedef c_decision_forest_classification_ModelBuilder::TreeId c_TreeId;

#define c_noParent c_decision_forest_classification_ModelBuilder::noParent

static daal::algorithms::decision_forest::classification::ModelPtr *
get_decision_forest_classification_modelbuilder_Model(daal::algorithms::decision_forest::classification::ModelBuilder * obj_)
{
    return RAW< daal::algorithms::decision_forest::classification::ModelPtr >()(obj_->getModel());
}
