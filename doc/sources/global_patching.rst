.. ******************************************************************************
.. * Copyright 2020-2021 Intel Corporation
.. *
.. * Licensed under the Apache License, Version 2.0 (the "License");
.. * you may not use this file except in compliance with the License.
.. * You may obtain a copy of the License at
.. *
.. *     http://www.apache.org/licenses/LICENSE-2.0
.. *
.. * Unless required by applicable law or agreed to in writing, software
.. * distributed under the License is distributed on an "AS IS" BASIS,
.. * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. * See the License for the specific language governing permissions and
.. * limitations under the License.
.. *******************************************************************************/

.. _global_patching:

###############
Global patching
###############

To patch your all scikit-learn applications for your environment without additional actions
you can use global patching.

You need **Scikit-learn** and **Intel(R) Extension for Scikit-learn** in your environment,
then use following command::

    python sklearnex.glob patch_sklearn

This command applies patching for all supported algorithms.

.. note::
    For correct work of global patching, you need read and write permissions to scikit-learn files.

If you need to patch only some algorithms (for example SVC and RandomForestClassifier estimators),
then use ``--algorithm`` or ``-a`` keys with list of needed algorithms::

    python sklearnex.glob patch_sklearn -a svc random_forest_classifier

If you don't want to receive patching notification, then use ``--no-verbose`` or ``-nv`` keys::

    python sklearnex.glob patch_sklearn -a svc random_forest_classifier -nv

.. note::
    If you run the global patching command several times with different parameters,
    then only the last configuration will be applied.

To disable global patching, use following command::

    python sklearnex.glob unpatch_sklearn

You can also enable global patching by your code, with helps ``global_patch``
argument in ``patch_sklearn`` function::

    from sklearnex import patch_sklearn
    patch_sklearn(global_patch=True)
    import sklearn

After that, sklearn patches will be enabled in the current application and
in all others using the same environment.

To disable global patching by code, use ``global_patch``
argument in ``unpatch_sklearn`` function::

    from sklearnex import unpatch_sklearn
    unpatch_sklearn(global_patch=True)

.. note::
    If you clone an environment with enabled global patching, it will already be applied in the new environment.
