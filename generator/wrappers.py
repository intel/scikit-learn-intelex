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

# Listing requried parameters for each algorithm.
# They are used to initialize the algorithm object instead of gettings set explicitly.
# Note: even though listed under 'Batch', they are currently also used for 'Distributed'
#  unless explicitly provided in a step spec.
required = {
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
}

# Some parameters/inputs are not used when C++ datastrcutures are shared across
# different algos (like training and prediction)
# List them here for the 'ignoring' algos.
ignore = {
    'algorithms::svm::training': ['weights'],
    'algorithms::multi_class_classifier::training': ['weights'],
    'algorithms::multinomial_naive_bayes::training': ['weights'],
    'algorithms::kmeans::init': ['nRowsTotal', 'offset',],
}

# List of InterFaces, classes that can be arguments to other algorithms
# Mapping iface name to fully qualified DAAL type as shared pointer
ifaces = {
    'kernel_function::KernelIface': 'daal::algorithms::kernel_function::KernelIfacePtr',
    'classifier::prediction::Batch': 'daal::services::SharedPtr<daal::algorithms::classifier::prediction::Batch>',
    'classifier::training::Batch': 'daal::services::SharedPtr<daal::algorithms::classifier::training::Batch>',
}

# By default input arguments have no default value (e.g. they are required).
# Here you can make input arguments and parameters optional by providing their
# default value (for each algorithm).
defaults = {
    'algorithms::multivariate_outlier_detection': {
        'location': 'data_management::NumericTablePtr()',
        'scatter': 'data_management::NumericTablePtr()',
        'threshold': 'data_management::NumericTablePtr()',
    },
    'algorithms::univariate_outlier_detection': {
        'location': 'data_management::NumericTablePtr()',
        'scatter': 'data_management::NumericTablePtr()',
        'threshold': 'data_management::NumericTablePtr()',
    },
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
                                 'inputnames',   # array of names of input args, aligned with 'input'
                             ]
)
SSpec.__new__.__defaults__ = (None,) * (len(SSpec._fields)-1) + (['data'],)

# We list all algos with distributed versions here.
# The indivdual dicts get passed to jinja as global vars (as-is).
# Each algorithm provides it distributed pattern and the configuration parameters.
# The latter are provided per step.
# Do we want to include the flow-spec in here?
has_dist = {
    'algorithms::pca' : {
        'pattern': 'applyGather',
        'step_specs': [SSpec(name      = 'step1Local',
                             input     = ['data_management::NumericTablePtr'],
                             output    = 'services::SharedPtr< algorithms::pca::PartialResult< method > >',
                             iomanager = 'PartialIOManager',
                             setinput  = ['algorithms::pca::data']),
                       SSpec(name      = 'step2Master',
                             input     = ['services::SharedPtr< algorithms::pca::PartialResult< method > >'],
                             output    = 'algorithms::pca::ResultPtr',
                             iomanager = 'IOManager',
                             addinput  = 'algorithms::pca::partialResults')
                   ],
    },
    'algorithms::multinomial_naive_bayes::training' : {
        'pattern': 'applyGather',
        'step_specs': [SSpec(name      = 'step1Local',
                             input      = ['data_management::NumericTablePtr', 'data_management::NumericTablePtr'],
                             output     = 'services::SharedPtr< algorithms::multinomial_naive_bayes::training::PartialResult >',
                             iomanager  = 'PartialIOManager2',
                             setinput   = ['algorithms::classifier::training::data', 'algorithms::classifier::training::labels'],
                             inputnames = ['data', 'labels']),
                       SSpec(name      = 'step2Master',
                             input      = ['services::SharedPtr< algorithms::multinomial_naive_bayes::training::PartialResult >'],
                             output     = 'algorithms::multinomial_naive_bayes::training::ResultPtr',
                             iomanager  = 'IOManager',
                             addinput   = 'algorithms::multinomial_naive_bayes::training::partialModels')
                   ],
    },
    'algorithms::linear_regression::training' : {
        'pattern': 'mapReduce',
        'step_specs': [SSpec(name      = 'step1Local',
                             input       = ['data_management::NumericTablePtr', 'data_management::NumericTablePtr'],
                             inputnames  = ['data', 'dependentVariables'],
                             output      = 'services::SharedPtr< algorithms::linear_regression::training::PartialResult >',
                             iomanager   = 'PartialIOManager2',
                             setinput    = ['algorithms::linear_regression::training::data', 'algorithms::linear_regression::training::dependentVariables'],),
                       SSpec(name      = 'step2Master',
                             input       = ['services::SharedPtr< algorithms::linear_regression::training::PartialResult >'],
                             output      = 'algorithms::linear_regression::training::PartialResultPtr',
                             iomanager   = 'PartialIOManager',
                             addinput    = 'algorithms::linear_regression::training::partialModels'),
                       SSpec(name      = 'step2Master__final',
                             input       = ['services::SharedPtr< algorithms::linear_regression::training::PartialResult >'],
                             output      = 'algorithms::linear_regression::training::ResultPtr',
                             iomanager   = 'IOManager',
                             addinput    = 'algorithms::linear_regression::training::partialModels')
                   ],
    },
    'algorithms::svd' : {
        'pattern': 'applyGather',
        'step_specs': [SSpec(name      = 'step1Local',
                             input     = ['data_management::NumericTablePtr'],
                             output    = 'data_management::DataCollectionPtr',
                             iomanager = 'PartialIOManagerSingle',
                             setinput  = ['algorithms::svd::data'],
                             iomargs   = ['algorithms::svd::PartialResultId', 'algorithms::svd::outputOfStep1ForStep2']),
                       SSpec(name      = 'step2Master',
                             input     = ['data_management::DataCollectionPtr'],
                             output    = 'algorithms::svd::ResultPtr',
                             iomanager = 'IOManager',
                             addinput  = 'algorithms::svd::inputOfStep2FromStep1, i')
                   ],
    },
    'algorithms::kmeans::init' : {
        #        'iombatch': 'IOManagerSingle< algob_type, services::SharedPtr< typename algob_type::InputType >, data_management::NumericTablePtr, algorithms::kmeans::init::ResultId, algorithms::kmeans::init::centroids >',
        'pattern': 'dkmi',
        'step_specs': [SSpec(name      = 'step1Local',
                             input     = ['data_management::NumericTablePtr'],
                             extrainput= 'size_t nRowsTotal, size_t offset',
                             setinput  = ['algorithms::kmeans::init::data'],
                             output    = 'algorithms::kmeans::init::PartialResultPtr',
                             iomanager = 'PartialIOManager',
                             inputnames = ['data'],
                             construct = '_nClusters, nRowsTotal, offset'),
                       SSpec(name      = 'step2Master',
                             input     = ['algorithms::kmeans::init::PartialResultPtr'],
                             output    = 'data_management::NumericTablePtr',
                             iomanager = 'IOManagerSingle',
                             iomargs   = ['algorithms::kmeans::init::ResultId', 'algorithms::kmeans::init::centroids'],
                             addinput  = 'algorithms::kmeans::init::partialResults'),
                       SSpec(name      = 'step2Local',
                             input     = ['data_management::NumericTablePtr', 'data_management::DataCollectionPtr', 'data_management::NumericTablePtr'],
                             setinput  = ['algorithms::kmeans::init::data', 'algorithms::kmeans::init::internalInput', 'algorithms::kmeans::init::inputOfStep2'],
                             inputnames = ['data', 'internalInput', 'inputOfStep2'],
                             output    = 'algorithms::kmeans::init::DistributedStep2LocalPlusPlusPartialResultPtr',
                             iomanager = 'PartialIOManager3',
                             construct = '_nClusters, input2 ? false : true'),
                       SSpec(name      = 'step3Master',
                             input     = ['data_management::NumericTablePtr'],
                             output    = 'algorithms::kmeans::init::DistributedStep3MasterPlusPlusPartialResultPtr',
                             iomanager = 'PartialIOManager',
                             addinput  = 'algorithms::kmeans::init::inputOfStep3FromStep2, i'),
                       SSpec(name      = 'step4Local',
                             input     = ['data_management::NumericTablePtr', 'data_management::DataCollectionPtr', 'data_management::NumericTablePtr'],
                             setinput  = ['algorithms::kmeans::init::data', 'algorithms::kmeans::init::internalInput', 'algorithms::kmeans::init::inputOfStep4FromStep3'],
                             inputnames = ['data', 'internalInput', 'inputOfStep4FromStep3'],
                             output    = 'data_management::NumericTablePtr',
                             iomanager = 'PartialIOManager3Single',
                             iomargs   = ['algorithms::kmeans::init::DistributedStep4LocalPlusPlusPartialResultId', 'algorithms::kmeans::init::outputOfStep4']),
                   ],
    },
    'algorithms::kmeans' : {
        'pattern': 'mapReduceIter',
        'step_specs': [SSpec(name      = 'step1Local',
                             input     = ['data_management::NumericTablePtr', 'data_management::NumericTablePtr'],
                             setinput  = ['algorithms::kmeans::data', 'algorithms::kmeans::inputCentroids'],
                             inputnames = ['data', 'inputCentroids'],
                             output    = 'algorithms::kmeans::PartialResultPtr',
                             iomanager = 'PartialIOManager2',
                             construct = '_nClusters, false',),
                       SSpec(name      = 'step2Master',
                             input     = ['algorithms::kmeans::PartialResultPtr'],
                             output    = 'algorithms::kmeans::PartialResultPtr',
                             iomanager = 'PartialIOManager',
                             addinput  = 'algorithms::kmeans::partialResults',
                             construct = '_nClusters',),
                       SSpec(name      = 'step2Master__final',
                             input     = ['algorithms::kmeans::PartialResultPtr'],
                             output    = 'algorithms::kmeans::ResultPtr',
                             iomanager = 'IOManager',
                             addinput  = 'algorithms::kmeans::partialResults',
                             construct = '_nClusters',)
                   ],
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
                    'template_decl': 'algorithms::linear_regression::prediction::Method',
                    'default': 'algorithms::linear_regression::prediction::defaultDense',
                    'values': ['algorithms::linear_regression::prediction::defaultDense',]
                }),
            ]),
            'specs': [
                {
                    'template_decl': ['fptype'],
                    'expl': OrderedDict([('method', 'algorithms::linear_regression::prediction::defaultDense')]),
                },
            ],
        },
    }
}
