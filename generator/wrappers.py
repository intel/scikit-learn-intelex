#*******************************************************************************
# Copyright 2014-2018 Intel Corporation
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

def get_algos(ver):
    algos = [
        'association_rules',
        'cholesky',
        'decision_forest',
        'decision_tree',
        'distance',
        'em_gmm',
        'engine',
        'gbt',
        'implicit_als',
        'kdtree_knn_classification',
        'kernel_function',
        'kmeans',
        'linear_regression',
        'logistic_regression',
        'math',
        'multi_class_classifier',
        'multinomial_naive_bayes',
        'normalization',
        'optimization_solver',
        'outlier_detection',
        'pca',
        'qr',
        'quantiles',
        'ridge_regression',
        'sorting',
        'stump::classification',  # stump itself is deprecated since 2019.2
        'stump::regression',
        'svd',
        'svm',
        'univariate_outlier_detection',
    ]
    # Note: boosting and weak_learner are deprecated since 2019.2, we just ignore them
    if ver > (2019, 1):
        algos += [
            'adaboost',
            'brownboost',
            'covariance',
            'logitboost',
            'moments',
        ]
    return algos

# Listing requried parameters for each algorithm.
# They are used to initialize the algorithm object instead of gettings set explicitly.
# Note: even though listed under 'Batch', they are currently also used for 'Distributed'
#  unless explicitly provided in a step spec.
required = {
    'algorithms::em_gmm': {
        'Batch': [('nComponents', 'size_t')],
    },
    'algorithms::em_gmm::init': {
        'Batch': [('nComponents', 'size_t')],
    },
    'algorithms::kmeans': {
        'Batch': [('nClusters', 'size_t'), ('maxIterations', 'size_t')],
    },
    'algorithms::kmeans::init': {
        'Batch': [('nClusters', 'size_t')],
    },
    'algorithms::multinomial_naive_bayes::training': {
        'Batch': [('nClasses', 'size_t')],
    },
    'algorithms::multinomial_naive_bayes::prediction': {
        'Batch': [('nClasses', 'size_t')],
    },
    'algorithms::multi_class_classifier::training': {
        'Batch': [('nClasses', 'size_t')],
    },
    'algorithms::multi_class_classifier::prediction': {
        'Batch': [('nClasses', 'size_t')],
    },
    'algorithms::gbt::classification::training': {
        'Batch': [('nClasses', 'size_t')],
    },
    'algorithms::gbt::classification::prediction': {
        'Batch': [('nClasses', 'size_t')],
    },
    'algorithms::logistic_regression::training': {
        'Batch': [('nClasses', 'size_t')],
    },
    'algorithms::logistic_regression::prediction': {
        'Batch': [('nClasses', 'size_t')],
    },
    'algorithms::decision_tree::classification::training': {
        'Batch': [('nClasses', 'size_t')],
    },
    'algorithms::decision_forest::classification::training': {
        'Batch': [('nClasses', 'size_t')],
    },
    'algorithms::decision_forest::classification::prediction': {
        'Batch': [('nClasses', 'size_t')],
    },
    'algorithms::logitboost::prediction': {
        'Batch': [('nClasses', 'size_t')],
    },
    'algorithms::logitboost::training': {
        'Batch': [('nClasses', 'size_t')],
    },
    'algorithms::optimization_solver::mse': {
        'Batch': [('numberOfTerms', 'size_t')],
    },
    'algorithms::optimization_solver::logistic_loss': {
        'Batch': [('numberOfTerms', 'size_t')],
    },
    'algorithms::optimization_solver::cross_entropy_loss': {
        'Batch': [('nClasses', 'size_t'), ('numberOfTerms', 'size_t')],
    },
    'algorithms::optimization_solver::sgd': {
        'Batch': [('function', 'daal::algorithms::optimization_solver::sum_of_functions::BatchPtr')],
    },
    'algorithms::optimization_solver::lbfgs': {
        'Batch': [('function', 'daal::algorithms::optimization_solver::sum_of_functions::BatchPtr')],
    },
    'algorithms::optimization_solver::adagrad': {
        'Batch': [('function', 'daal::algorithms::optimization_solver::sum_of_functions::BatchPtr')],
    },
}

# Some algorithms have no public constructors and need to be instantiated with 'create'
# (for whatever reason)
no_constructor = {
    'algorithms::engines::mt19937::Batch': {'seed': 'size_t'},
    'algorithms::engines::mt2203::Batch': {'seed': 'size_t'},
    'algorithms::engines::mcg59::Batch': {'seed': 'size_t'},
}

# Some algorithms require a setup function, to provide input without actual compute
add_setup = [
    'algorithms::optimization_solver::mse',
    'algorithms::optimization_solver::logistic_loss',
    'algorithms::optimization_solver::cross_entropy_loss',
]

# Some parameters/inputs are not used when C++ datastructures are shared across
# different algos (like training and prediction)
# List them here for the 'ignoring' algos.
# Also lists input set/gets to ignore
ignore = {
    'algorithms::adaboost::training': ['weights'],
    'algorithms::brownboost::training': ['weights'],
    'algorithms::logitboost::training': ['weights'],
    'algorithms::svm::training': ['weights'],
    'algorithms::kdtree_knn_classification::training': ['weights'],
    'algorithms::multi_class_classifier::training': ['weights'],
    'algorithms::multinomial_naive_bayes::training': ['weights'],
    'algorithms::kmeans::init': ['nRowsTotal', 'offset',],
    'algorithms::gbt::regression::training': ['dependentVariables', 'weights'],
    'algorithms::gbt::classification::training': ['weights',],
    'algorithms::logistic_regression::training': ['weights',],
    'algorithms::decision_tree::classification::training': ['weights',],
    'algorithms::decision_tree::regression::training': ['weights',],
    'algorithms::decision_forest::classification::training': ['weights', 'updatedEngine',],
    'algorithms::decision_forest::regression::training': ['algorithms::regression::training::InputId', 'updatedEngine',],
    'algorithms::linear_regression::training': ['weights',],
    'algorithms::linear_regression::prediction': ['algorithms::linear_model::interceptFlag',],
    'algorithms::ridge_regression::training': ['weights',],
    'algorithms::ridge_regression::prediction': ['algorithms::linear_model::interceptFlag',],
    'algorithms::optimization_solver::sgd': ['optionalArgument', 'algorithms::optimization_solver::iterative_solver::OptionalResultId',
                                             'pastUpdateVector', 'pastWorkValue'],
    'algorithms::optimization_solver::lbfgs': ['optionalArgument', 'algorithms::optimization_solver::iterative_solver::OptionalResultId',
                                               'correctionPairs', 'correctionIndices', 'averageArgumentLIterations',],
    'algorithms::optimization_solver::adagrad': ['optionalArgument', 'algorithms::optimization_solver::iterative_solver::OptionalResultId',
                                                 'gradientSquareSum'],
    'algorithms::optimization_solver::objective_function': [],
    'algorithms::optimization_solver::iterative_solver': [],
    'algorithms::normalization::minmax': ['moments'],
    'algorithms::em_gmm': ['inputValues', 'covariance'], # optional input, parameter
    'algorithms::pca': ['correlation'],
    'algorithms::stump::classification::training': ['weights'],
    'algorithms::stump::regression::training': ['weights'],
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
}

# By default input arguments have no default value (e.g. they are required).
# Here you can make input arguments and parameters optional by providing their
# default value (for each algorithm).
defaults = {
    'algorithms::multivariate_outlier_detection': {
        'location': 'daal::data_management::NumericTablePtr()',
        'scatter': 'daal::data_management::NumericTablePtr()',
        'threshold': 'daal::data_management::NumericTablePtr()',
    },
    'algorithms::univariate_outlier_detection': {
        'location': 'daal::data_management::NumericTablePtr()',
        'scatter': 'daal::data_management::NumericTablePtr()',
        'threshold': 'daal::data_management::NumericTablePtr()',
    },
}

# In some cases we the whole C++ business is too complex and we need to provide
# types manually.
fallbacks = {
    'algorithms::pca': {
        'ParameterType': 'algorithms::pca::BaseBatchParameter',
    }
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
    'algorithms::cholesky::Batch': ['ParameterType',],
    'algorithms::classifier': ['Result',],
    'algorithms::correlation_distance::Batch': ['ParameterType',],
    'algorithms::cosine_distance::Batch': ['ParameterType',],
    'algorithms::decision_forest': ['Result',],
    'algorithms::decision_forest::classification': ['Result',],
    'algorithms::decision_forest::regression': ['Result',],
    'algorithms::decision_forest::regression::prediction::Batch': ['ParameterType',],
    'algorithms::decision_forest::training': ['Result',],
    'algorithms::decision_tree': ['Result',],
    'algorithms::decision_tree::classification': ['Result',],
    'algorithms::decision_tree::regression': ['Result',],
    'algorithms::engines::mcg59::Batch': ['ParameterType',],
    'algorithms::engines::mt19937::Batch': ['ParameterType',],
    'algorithms::engines::mt2203::Batch': ['ParameterType',],
    'algorithms::gbt': ['Result',],
    'algorithms::gbt::classification': ['Result',],
    'algorithms::gbt::regression': ['Result',],
    'algorithms::gbt::training': ['Result',],
    'algorithms::implicit_als': ['Result',],
    'algorithms::implicit_als::prediction': ['Result',],
    'algorithms::kdtree_knn_classification': ['Result',],
    'algorithms::linear_model': ['Result',],
    'algorithms::linear_regression': ['Result',],
    'algorithms::linear_regression::prediction::Batch': ['ParameterType',],
    'algorithms::logistic_regression': ['Result',],
    'algorithms::math': ['Result',],
    'algorithms::math::abs::Batch': ['ParameterType',],
    'algorithms::math::logistic::Batch': ['ParameterType',],
    'algorithms::math::relu::Batch': ['ParameterType',],
    'algorithms::math::smoothrelu::Batch': ['ParameterType',],
    'algorithms::math::softmax::Batch': ['ParameterType',],
    'algorithms::math::tanh::Batch': ['ParameterType',],
    'algorithms::multi_class_classifier': ['Result',],
    'algorithms::multinomial_naive_bayes': ['Result',],
    'algorithms::multivariate_outlier_detection::Batch': ['ParameterType',],
    'algorithms::normalization': ['Result',],
    'algorithms::normalization::zscore::Batch': ['ParameterType',],
    'algorithms::logitboost': ['Result',],
    'algorithms::optimization_solver': ['Result',],
    'algorithms::qr::Batch': ['ParameterType',],
    'algorithms::regression': ['Result',],
    'algorithms::ridge_regression': ['Result',],
    'algorithms::ridge_regression::prediction::Batch': ['ParameterType',],
    'algorithms::sorting::Batch': ['ParameterType',],
    'algorithms::svm': ['Result',],
    'algorithms::stump::classification': ['Result',],
    'algorithms::stump::regression': ['Result',],
    'algorithms::stump::regression::prediction::Batch': ['ParameterType',],
    'algorithms::univariate_outlier_detection::Batch': ['ParameterType',],
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
