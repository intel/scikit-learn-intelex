#*******************************************************************************
# Copyright 2014-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#******************************************************************************/

from collections import defaultdict, OrderedDict, namedtuple

# given a C++ namespace and a DAAL version, return if namespace/algo should be
# wrapped in daal4py.
def wrap_algo(algo, ver):
    #return True if 'kmeans' in algo and not 'interface' in algo else False
    # Ignore some algos if using older DAAL
    if ver < (2019, 5) and any(x in algo for x in ['stump', 'adaboost', 'brownboost', 'logitboost',]):
        return False
    # ignore deprecated version of stump
    if 'stump' in algo and not any(x in algo for x in ['stump::regression', 'stump::classification']):
        return False
    # other deprecated algos
    if any(x in algo for x in ['boosting', 'weak_learner']):
        return False
    # ignore unneeded stuff
    if any(algo.endswith(x) for x in ['daal', 'algorithms',
                                      'algorithms::linear_model::prediction', 'algorithms::linear_model::training',
                                      'algorithms::classification::prediction', 'algorithms::classification::training',
                                      'algorithms::tree_utils', 'algorithms::tree_utils::classification', 'algorithms::tree_utils::regression',]):
        return False
    # ignore unsupported algos
    if any(x in algo for x in ['quality_metric', '::interface']):
        return False

    return True


# Listing required parameters for each algorithm.
# They are used to initialize the algorithm object instead of gettings set explicitly.
# Note: even though listed under 'Batch', they are currently also used for 'Distributed'
#  unless explicitly provided in a step spec.
required = {
    'algorithms::distributions::bernoulli': [('p', 'double')],
    'algorithms::em_gmm': [('nComponents', 'size_t')],
    'algorithms::em_gmm::init': [('nComponents', 'size_t')],
    'algorithms::kmeans': [('nClusters', 'size_t'), ('maxIterations', 'size_t')],
    'algorithms::kmeans::init': [('nClusters', 'size_t')],
    'algorithms::multinomial_naive_bayes::training': [('nClasses', 'size_t')],
    'algorithms::multinomial_naive_bayes::prediction': [('nClasses', 'size_t')],
    'algorithms::multi_class_classifier::training': [('nClasses', 'size_t')],
    'algorithms::multi_class_classifier::prediction': [('nClasses', 'size_t')],
    'algorithms::gbt::classification::training': [('nClasses', 'size_t')],
    'algorithms::gbt::classification::prediction': [('nClasses', 'size_t')],
    'algorithms::logistic_regression::training': [('nClasses', 'size_t')],
    'algorithms::logistic_regression::prediction': [('nClasses', 'size_t')],
    'algorithms::decision_tree::classification::training': [('nClasses', 'size_t')],
    'algorithms::decision_forest::classification::training': [('nClasses', 'size_t')],
    'algorithms::decision_forest::classification::prediction': [('nClasses', 'size_t')],
    'algorithms::logitboost::prediction': [('nClasses', 'size_t')],
    'algorithms::logitboost::training': [('nClasses', 'size_t')],
    'algorithms::optimization_solver::mse': [('numberOfTerms', 'size_t')],
    'algorithms::optimization_solver::logistic_loss': [('numberOfTerms', 'size_t')],
    'algorithms::optimization_solver::cross_entropy_loss': [('nClasses', 'size_t'), ('numberOfTerms', 'size_t')],
    'algorithms::optimization_solver::sgd': [('function', 'daal::algorithms::optimization_solver::sum_of_functions::BatchPtr')],
    'algorithms::optimization_solver::lbfgs': [('function', 'daal::algorithms::optimization_solver::sum_of_functions::BatchPtr')],
    'algorithms::optimization_solver::adagrad': [('function', 'daal::algorithms::optimization_solver::sum_of_functions::BatchPtr')],
}

# Some algorithms have no public constructors and need to be instantiated with 'create'
# (for whatever reason)
no_constructor = {
    'algorithms::engines::mt19937::Batch': {'seed': ['size_t', 'seed']},
    'algorithms::engines::mt2203::Batch': {'seed': ['size_t', 'seed']},
    'algorithms::engines::mcg59::Batch': {'seed': ['size_t', 'seed']},
}

# Some algorithms require a setup function, to provide input without actual compute
# Also need to add it to doc/algorithms.rst
add_setup = [
    'algorithms::optimization_solver::mse',
    'algorithms::optimization_solver::logistic_loss',
    'algorithms::optimization_solver::cross_entropy_loss',
]

# Some parameters/inputs are not used when C++ datastructures are shared across
# different algos (like training and prediction)
# List them here for the 'ignoring' algos.
# Also lists input set/gets to ignore
# Adding an empty list to ignore all parameters, inputs and results
ignore = {
    'algorithms::kmeans::init': ['firstIteration', 'outputForStep5Required',], # internal for distributed
    'algorithms::kmeans::init::interface1': ['nRowsTotal', 'offset', 'seed',], # internal for distributed, deprecated
    'algorithms::gbt::regression::training': ['dependentVariables'], # dependentVariables from parent class is not used
    'algorithms::decision_forest::training': ['seed',], # deprecated
    'algorithms::decision_forest::classification::training': ['updatedEngine',], # output
    'algorithms::decision_forest::regression::training': ['algorithms::regression::training::InputId', # InputId from parent class is not used
                                                          'updatedEngine',], # output
    'algorithms::linear_regression::prediction': ['algorithms::linear_model::interceptFlag',], # parameter
    'algorithms::ridge_regression::prediction': ['algorithms::linear_model::interceptFlag',], # parameter
    'algorithms::optimization_solver::sgd': ['optionalArgument', 'algorithms::optimization_solver::iterative_solver::OptionalResultId',
                                             'pastUpdateVector', 'pastWorkValue', 'seed',], # internal stuff, deprecated
    'algorithms::optimization_solver::lbfgs': ['optionalArgument', 'algorithms::optimization_solver::iterative_solver::OptionalResultId',
                                               'correctionPairs', 'correctionIndices', 'averageArgumentLIterations', 'seed',], # internal stuff, deprecated
    'algorithms::optimization_solver::adagrad': ['optionalArgument', 'algorithms::optimization_solver::iterative_solver::OptionalResultId',
                                                 'gradientSquareSum', 'seed',], # internal stuff, deprecated
    'algorithms::optimization_solver::saga': ['optionalArgument', 'algorithms::optimization_solver::iterative_solver::OptionalResultId', 'seed',], # internal stuff, deprecated
    'algorithms::optimization_solver::objective_function': [], # interface type
    'algorithms::optimization_solver::iterative_solver': [], # interface type
    'algorithms::normalization::minmax': ['moments'], # parameter, required an interface
    'algorithms::normalization::zscore': ['moments'], # parameter, required an interface
    'algorithms::em_gmm': ['inputValues', 'covariance'], # optional input is dictionary, parameter
    'algorithms::em_gmm::init': ['seed',], # deprecated
    'algorithms::pca': ['covariance'], # parameter defined multiple times with different types
    'algorithms::kdtree_knn_classification': ['seed',], # deprecated
}

# List of InterFaces, classes that can be arguments to other algorithms
# Mapping iface class to fully qualified DAAL type as shared pointer
ifaces = {
    'kernel_function::KernelIface': ('daal::algorithms::kernel_function::KernelIfacePtr', None),
    'classifier::prediction::Batch': ('daal::services::SharedPtr<daal::algorithms::classifier::prediction::Batch>', None),
    'classifier::training::Batch': ('daal::services::SharedPtr<daal::algorithms::classifier::training::Batch>', None),
    'engines::BatchBase': ('daal::algorithms::engines::EnginePtr', None),
    'engines::FamilyBatchBase': ('daal::algorithms::engines::EnginePtr', 'daal::algorithms::engines::EnginePtr'),
    'optimization_solver::sum_of_functions::Batch': ('daal::algorithms::optimization_solver::sum_of_functions::BatchPtr', None),
    'optimization_solver::iterative_solver::Batch': ('daal::algorithms::optimization_solver::iterative_solver::BatchPtr', None),
    'regression::training::Batch': ('daal::services::SharedPtr<daal::algorithms::regression::training::Batch>', None),
    'regression::prediction::Batch': ('daal::services::SharedPtr<daal::algorithms::regression::prediction::Batch>', None),
    'normalization::zscore::BatchImpl': ('daal::services::SharedPtr<daal::algorithms::normalization::zscore::BatchImpl>', None),
}

# By default input arguments have no default value (e.g. they are required).
# Here you can make input arguments and parameters optional by providing their
# default value (for each algorithm).
# Set True value for standart data types and shared pointers.
defaults = {
    'algorithms::pca': {'correlation': True,},
    'algorithms::multivariate_outlier_detection': {'location': True, 'scatter': True, 'threshold': True,},
    'algorithms::univariate_outlier_detection': {'location': True, 'scatter': True, 'threshold': True,},
    'algorithms::adaboost::training': {'weights': True,},
    'algorithms::brownboost::training': {'weights': True,},
    'algorithms::logitboost::training': {'weights': True,},
    'algorithms::svm::training': {'weights': True,},
    'algorithms::kdtree_knn_classification::training': {'weights': True,},
    'algorithms::multi_class_classifier::training': {'weights': True,},
    'algorithms::multinomial_naive_bayes::training': {'weights': True,},
    'algorithms::gbt::regression::training': {'weights': True,},
    'algorithms::gbt::classification::training': {'weights': True,},
    'algorithms::logistic_regression::training': {'weights': True,},
    'algorithms::decision_tree::classification::training': {'weights': True,},
    'algorithms::decision_tree::regression::training': {'weights': True,},
    'algorithms::decision_forest::classification::training': {'weights': True,},
    'algorithms::linear_regression::training': {'weights': True,},
    'algorithms::ridge_regression::training': {'weights': True,},
    'algorithms::stump::classification::training': {'weights': True,},
    'algorithms::stump::regression::training': {'weights': True,},
}

# For enums that are used to access KeyValueDataCollections we need an inverse map
# value->string.
enum_maps = {
    'algorithms::pca::ResultToComputeId' : 'result_dataForTransform',
}

# The distributed algorithm configuration parameters
# Note that all have defaults and so are optional.
# In particular note that the name of a single input argument defaults to data.
# Use the DAAL step enums as names. You can use the same enum value by prepending '__*' to the name.
#  Trailing '__*' will be trunkated when using the name as a template argument.
SSpec = namedtuple('step_spec', ['input',        # array of input types
                                 'extrainput',   # extra input arguments (added to run_* as-is)
                                 'output',       # output type
                                 'iomanager',    # IOManager with typedefs and result access
                                 'setinput',     # array of enum-values to set inputs, aligned with 'input'
                                 'addinput',     # arguments to adding input (step2 and after)
                                 'iomargs',      # arguments to IOManager
                                 'staticinput',  # array of inputs that come from user and are unpartitioned
                                 'name',         # step1Local, step2Local, step3Master, ...
                                 'construct',    # args to algo constructor if non-default
                                 'params',       # indicates if init_parameters should be called
                                 'inputnames',   # array of names of input args, aligned with 'input'
                                 'inputdists',   # array of distributions (hpat) of input args, aligned with 'input'
                             ]
)
SSpec.__new__.__defaults__ = (None,) * (len(SSpec._fields)-3) + (True, ['data'], ['OneD'])

# We list all algos with distributed versions here.
# The indivdual dicts get passed to jinja as global vars (as-is).
# Each algorithm provides it distributed pattern and the configuration parameters.
# The latter are provided per step.
# Do we want to include the flow-spec in here?
has_dist = {
    'algorithms::pca' : {
        'pattern': 'map_reduce_star',
        'step_specs': [SSpec(name      = 'step1Local',
                             input     = ['daal::data_management::NumericTablePtr'],
                             output    = 'daal::services::SharedPtr< daal::algorithms::pca::PartialResult< method > >',
                             iomanager = 'PartialIOManager',
                             setinput  = ['daal::algorithms::pca::data'],
                             params    = False),
                       SSpec(name      = 'step2Master',
                             input     = ['daal::services::SharedPtr< daal::algorithms::pca::PartialResult< method > >'],
                             output    = 'daal::algorithms::pca::ResultPtr',
                             iomanager = 'IOManager',
                             addinput  = 'daal::algorithms::pca::partialResults',
                             params    = False)
                   ],
    },
    'algorithms::low_order_moments' : {
        'pattern': 'map_reduce_tree',
        'step_specs': [SSpec(name      = 'step1Local',
                             input     = ['daal::data_management::NumericTablePtr'],
                             output    = 'daal::services::SharedPtr< daal::algorithms::low_order_moments::PartialResult >',
                             iomanager = 'PartialIOManager',
                             setinput  = ['daal::algorithms::low_order_moments::data']),
                       SSpec(name      = 'step2Master',
                             input     = ['daal::services::SharedPtr< daal::algorithms::low_order_moments::PartialResult >'],
                             output    = 'daal::algorithms::low_order_moments::PartialResultPtr',
                             iomanager = 'PartialIOManager',
                             addinput  = 'daal::algorithms::low_order_moments::partialResults'),
                       SSpec(name      = 'step2Master__final',
                             input     = ['daal::services::SharedPtr< daal::algorithms::low_order_moments::PartialResult >'],
                             output    = 'daal::algorithms::low_order_moments::ResultPtr',
                             iomanager = 'IOManager',
                             addinput  = 'daal::algorithms::low_order_moments::partialResults')
                   ],
    },
    'algorithms::covariance' : {
        'pattern': 'map_reduce_tree',
        'step_specs': [SSpec(name      = 'step1Local',
                             input     = ['daal::data_management::NumericTablePtr'],
                             output    = 'daal::services::SharedPtr< daal::algorithms::covariance::PartialResult >',
                             iomanager = 'PartialIOManager',
                             setinput  = ['daal::algorithms::covariance::data']),
                       SSpec(name      = 'step2Master',
                             input     = ['daal::services::SharedPtr< daal::algorithms::covariance::PartialResult >'],
                             output    = 'daal::algorithms::covariance::PartialResultPtr',
                             iomanager = 'PartialIOManager',
                             addinput  = 'daal::algorithms::covariance::partialResults'),
                       SSpec(name      = 'step2Master__final',
                             input     = ['daal::services::SharedPtr< daal::algorithms::covariance::PartialResult >'],
                             output    = 'daal::algorithms::covariance::ResultPtr',
                             iomanager = 'IOManager',
                             addinput  = 'daal::algorithms::covariance::partialResults')
                   ],
    },
    'algorithms::multinomial_naive_bayes::training' : {
        'pattern': 'map_reduce_star',
        'step_specs': [SSpec(name      = 'step1Local',
                             input      = ['daal::data_management::NumericTablePtr', 'daal::data_management::NumericTablePtr'],
                             output     = 'daal::services::SharedPtr< daal::algorithms::multinomial_naive_bayes::training::PartialResult >',
                             iomanager  = 'PartialIOManager2',
                             setinput   = ['daal::algorithms::classifier::training::data', 'daal::algorithms::classifier::training::labels'],
                             inputnames = ['data', 'labels'],
                             inputdists = ['OneD', 'OneD']),
                       SSpec(name      = 'step2Master',
                             input      = ['daal::services::SharedPtr< daal::algorithms::multinomial_naive_bayes::training::PartialResult >'],
                             output     = 'daal::algorithms::multinomial_naive_bayes::training::ResultPtr',
                             iomanager  = 'IOManager',
                             addinput   = 'daal::algorithms::multinomial_naive_bayes::training::partialModels')
                   ],
    },
    'algorithms::linear_regression::training' : {
        'pattern': 'map_reduce_tree',
        'step_specs': [SSpec(name      = 'step1Local',
                             input       = ['daal::data_management::NumericTablePtr', 'daal::data_management::NumericTablePtr'],
                             inputnames  = ['data', 'dependentVariables'],
                             inputdists  = ['OneD', 'OneD'],
                             output      = 'daal::services::SharedPtr< daal::algorithms::linear_regression::training::PartialResult >',
                             iomanager   = 'PartialIOManager2',
                             setinput    = ['daal::algorithms::linear_regression::training::data', 'daal::algorithms::linear_regression::training::dependentVariables'],),
                       SSpec(name      = 'step2Master',
                             input       = ['daal::services::SharedPtr< daal::algorithms::linear_regression::training::PartialResult >'],
                             output      = 'daal::algorithms::linear_regression::training::PartialResultPtr',
                             iomanager   = 'PartialIOManager',
                             addinput    = 'daal::algorithms::linear_regression::training::partialModels'),
                       SSpec(name      = 'step2Master__final',
                             input       = ['daal::services::SharedPtr< daal::algorithms::linear_regression::training::PartialResult >'],
                             output      = 'daal::algorithms::linear_regression::training::ResultPtr',
                             iomanager   = 'IOManager',
                             addinput    = 'daal::algorithms::linear_regression::training::partialModels')
                   ],
    },
    'algorithms::ridge_regression::training' : {
        'pattern': 'map_reduce_tree',
        'step_specs': [SSpec(name      = 'step1Local',
                             input       = ['daal::data_management::NumericTablePtr', 'daal::data_management::NumericTablePtr'],
                             inputnames  = ['data', 'dependentVariables'],
                             inputdists  = ['OneD', 'OneD'],
                             output      = 'daal::services::SharedPtr< daal::algorithms::ridge_regression::training::PartialResult >',
                             iomanager   = 'PartialIOManager2',
                             setinput    = ['daal::algorithms::ridge_regression::training::data', 'daal::algorithms::ridge_regression::training::dependentVariables'],),
                       SSpec(name      = 'step2Master',
                             input       = ['daal::services::SharedPtr< daal::algorithms::ridge_regression::training::PartialResult >'],
                             output      = 'daal::algorithms::ridge_regression::training::PartialResultPtr',
                             iomanager   = 'PartialIOManager',
                             addinput    = 'daal::algorithms::ridge_regression::training::partialModels'),
                       SSpec(name      = 'step2Master__final',
                             input       = ['daal::services::SharedPtr< daal::algorithms::ridge_regression::training::PartialResult >'],
                             output      = 'daal::algorithms::ridge_regression::training::ResultPtr',
                             iomanager   = 'IOManager',
                             addinput    = 'daal::algorithms::ridge_regression::training::partialModels')
                   ],
    },
    'algorithms::svd' : {
        'pattern': 'map_reduce_star',
        'step_specs': [SSpec(name      = 'step1Local',
                             input     = ['daal::data_management::NumericTablePtr'],
                             output    = 'daal::data_management::DataCollectionPtr',
                             iomanager = 'PartialIOManagerSingle',
                             setinput  = ['daal::algorithms::svd::data'],
                             iomargs   = ['daal::algorithms::svd::PartialResultId', 'daal::algorithms::svd::outputOfStep1ForStep2']),
                       SSpec(name      = 'step2Master',
                             input     = ['daal::data_management::DataCollectionPtr'],
                             output    = 'daal::algorithms::svd::ResultPtr',
                             iomanager = 'IOManager',
                             addinput  = 'daal::algorithms::svd::inputOfStep2FromStep1, i')
                   ],
    },
    'algorithms::kmeans::init' : {
        #        'iombatch': 'IOManagerSingle< algob_type, daal::services::SharedPtr< typename algob_type::InputType >, daal::data_management::NumericTablePtr, daal::algorithms::kmeans::init::ResultId, daal::algorithms::kmeans::init::centroids >',
        'pattern': 'dist_custom',
        'step_specs': [SSpec(name      = 'step1Local',
                             input     = ['daal::data_management::NumericTablePtr'],
                             extrainput= 'size_t nRowsTotal, size_t offset',
                             setinput  = ['daal::algorithms::kmeans::init::data'],
                             output    = 'daal::algorithms::kmeans::init::PartialResultPtr',
                             iomanager = 'PartialIOManager',
                             construct = '_nClusters, nRowsTotal, offset'),
                       SSpec(name      = 'step2Master',
                             input     = ['daal::algorithms::kmeans::init::PartialResultPtr'],
                             output    = 'daal::data_management::NumericTablePtr',
                             iomanager = 'IOManagerSingle',
                             iomargs   = ['daal::algorithms::kmeans::init::ResultId', 'daal::algorithms::kmeans::init::centroids'],
                             addinput  = 'daal::algorithms::kmeans::init::partialResults'),
                       SSpec(name      = 'step2Local',
                             input     = ['daal::data_management::NumericTablePtr', 'daal::data_management::DataCollectionPtr', 'daal::data_management::NumericTablePtr'],
                             setinput  = ['daal::algorithms::kmeans::init::data', 'daal::algorithms::kmeans::init::internalInput', 'daal::algorithms::kmeans::init::inputOfStep2'],
                             inputnames = ['data', 'internalInput', 'inputOfStep2'],
                             output    = 'daal::algorithms::kmeans::init::DistributedStep2LocalPlusPlusPartialResultPtr',
                             iomanager = 'PartialIOManager3',
                             construct = '_nClusters, input2 ? false : true'),
                       SSpec(name      = 'step3Master',
                             input     = ['daal::data_management::NumericTablePtr'],
                             output    = 'daal::algorithms::kmeans::init::DistributedStep3MasterPlusPlusPartialResultPtr',
                             iomanager = 'PartialIOManager',
                             addinput  = 'daal::algorithms::kmeans::init::inputOfStep3FromStep2, i'),
                       SSpec(name      = 'step4Local',
                             input     = ['daal::data_management::NumericTablePtr', 'daal::data_management::DataCollectionPtr', 'daal::data_management::NumericTablePtr'],
                             setinput  = ['daal::algorithms::kmeans::init::data', 'daal::algorithms::kmeans::init::internalInput', 'daal::algorithms::kmeans::init::inputOfStep4FromStep3'],
                             inputnames = ['data', 'internalInput', 'inputOfStep4FromStep3'],
                             output    = 'daal::data_management::NumericTablePtr',
                             iomanager = 'PartialIOManager3Single',
                             iomargs   = ['daal::algorithms::kmeans::init::DistributedStep4LocalPlusPlusPartialResultId', 'daal::algorithms::kmeans::init::outputOfStep4']),
                   ],
    },
    'algorithms::kmeans' : {
        'pattern': 'dist_custom',
        'step_specs': [SSpec(name      = 'step1Local',
                             input     = ['daal::data_management::NumericTablePtr', 'daal::data_management::NumericTablePtr'],
                             setinput  = ['daal::algorithms::kmeans::data', 'daal::algorithms::kmeans::inputCentroids'],
                             inputnames = ['data', 'inputCentroids'],
                             inputdists  = ['OneD', 'REP'],
                             output    = 'daal::algorithms::kmeans::PartialResultPtr',
                             iomanager = 'PartialIOManager2',
                             construct = '_nClusters, false',),
                       SSpec(name      = 'step2Master',
                             input     = ['daal::algorithms::kmeans::PartialResultPtr'],
                             output    = 'daal::algorithms::kmeans::PartialResultPtr',
                             iomanager = 'PartialIOManager',
                             addinput  = 'daal::algorithms::kmeans::partialResults',
                             construct = '_nClusters',),
                       SSpec(name      = 'step2Master__final',
                             input     = ['daal::algorithms::kmeans::PartialResultPtr'],
                             output    = 'daal::algorithms::kmeans::ResultPtr',
                             iomanager = 'IOManager',
                             addinput  = 'daal::algorithms::kmeans::partialResults',
                             construct = '_nClusters',)
                   ],
    },
    'algorithms::logistic_regression::training' : {
        'pattern': 'dist_custom',
        'step_specs' : [],
        'inputnames' : ['data', 'labels'],
        'inputdists' : ['OneD', 'OneD'],
    },
}

# Algorithms might have explicitly specializations in which input/ouput/parameter types
# are also specialized. Currently we only support this for the parameter type.
# See older version of wrappers.py or swig_interface.py for the expected syntax.
specialized = {
    'algorithms::linear_regression::prediction': {
        'Batch': {
            'tmpl_decl': OrderedDict([
                ('fptype', {
                    'template_decl': 'typename',
                    'default': 'double',
                    'values': ['double', 'float']
                }),
                ('method', {
                    'template_decl': 'daal::algorithms::linear_regression::prediction::Method',
                    'default': 'daal::algorithms::linear_regression::prediction::defaultDense',
                    'values': ['daal::algorithms::linear_regression::prediction::defaultDense',]
                }),
            ]),
            'specs': [
                {
                    'template_decl': ['fptype'],
                    'expl': OrderedDict([('method', 'daal::algorithms::linear_regression::prediction::defaultDense')]),
                },
            ],
        },
    },
    'algorithms::ridge_regression::prediction': {
        'Batch': {
            'tmpl_decl': OrderedDict([
                ('fptype', {
                    'template_decl': 'typename',
                    'default': 'double',
                    'values': ['double', 'float']
                }),
                ('method', {
                    'template_decl': 'daal::algorithms::ridge_regression::prediction::Method',
                    'default': 'daal::algorithms::ridge_regression::prediction::defaultDense',
                    'values': ['daal::algorithms::ridge_regression::prediction::defaultDense',]
                }),
            ]),
            'specs': [
                {
                    'template_decl': ['fptype'],
                    'expl': OrderedDict([('method', 'daal::algorithms::ridge_regression::prediction::defaultDense')]),
                },
            ],
        },
    },
}

no_warn = {
    'algorithms::adaboost': ['Result',],
    'algorithms::boosting': ['Result',],
    'algorithms::brownboost': ['Result',],
    'algorithms::cholesky': ['ParameterType',],
    'algorithms::classifier': ['Result',],
    'algorithms::correlation_distance': ['ParameterType',],
    'algorithms::cosine_distance': ['ParameterType',],
    'algorithms::decision_forest': ['Result',],
    'algorithms::decision_forest::classification': ['Result',],
    'algorithms::decision_forest::regression': ['Result',],
    'algorithms::decision_forest::regression::prediction': ['ParameterType',],
    'algorithms::decision_forest::training': ['Result',],
    'algorithms::decision_tree': ['Result',],
    'algorithms::decision_tree::classification': ['Result',],
    'algorithms::decision_tree::regression': ['Result',],
    'algorithms::engines::mcg59': ['ParameterType',],
    'algorithms::engines::mt19937': ['ParameterType',],
    'algorithms::engines::mt2203': ['ParameterType',],
    'algorithms::gbt': ['Result',],
    'algorithms::gbt::classification': ['Result',],
    'algorithms::gbt::regression': ['Result',],
    'algorithms::gbt::training': ['Result',],
    'algorithms::implicit_als': ['Result',],
    'algorithms::implicit_als::prediction': ['Result',],
    'algorithms::kdtree_knn_classification': ['Result',],
    'algorithms::linear_model': ['Result',],
    'algorithms::linear_regression': ['Result',],
    'algorithms::linear_regression::prediction': ['ParameterType',],
    'algorithms::logistic_regression': ['Result',],
    'algorithms::math': ['Result',],
    'algorithms::math::abs': ['ParameterType',],
    'algorithms::math::logistic': ['ParameterType',],
    'algorithms::math::relu': ['ParameterType',],
    'algorithms::math::smoothrelu': ['ParameterType',],
    'algorithms::math::softmax': ['ParameterType',],
    'algorithms::math::tanh': ['ParameterType',],
    'algorithms::multi_class_classifier': ['Result',],
    'algorithms::multinomial_naive_bayes': ['Result',],
    'algorithms::multivariate_outlier_detection': ['ParameterType',],
    'algorithms::normalization': ['Result',],
    'algorithms::normalization::zscore': ['ParameterType',],
    'algorithms::logitboost': ['Result',],
    'algorithms::optimization_solver': ['Result',],
    'algorithms::qr': ['ParameterType',],
    'algorithms::regression': ['Result',],
    'algorithms::ridge_regression': ['Result',],
    'algorithms::ridge_regression::prediction': ['ParameterType',],
    'algorithms::sorting': ['ParameterType',],
    'algorithms::svm': ['Result',],
    'algorithms::stump::classification': ['Result',],
    'algorithms::stump::regression': ['Result',],
    'algorithms::stump::regression::prediction': ['ParameterType',],
    'algorithms::univariate_outlier_detection': ['ParameterType',],
    'algorithms::weak_learner': ['Result',],
}

# we need to be more specific about numeric table types for the lowering phase in HPAT
# We provide specific types here
hpat_types = {
    'kmeans_result': {
        'assignments': 'itable_type',
        'nIterations': 'itable_type',
    },
}
