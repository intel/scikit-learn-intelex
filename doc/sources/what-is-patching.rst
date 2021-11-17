.. ******************************************************************************
.. * Copyright 2021 Intel Corporation
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


########
Patching
########

.. glossary::

   patching
      To patch scikit-learn with |intelex| is to replace stock scikit-learn algorithms
      with their optimized versions provided by the extension.

      There are different ways to patch scikit-learn:

      .. include:: /patching/patching-options.rst

      .. seealso:: :ref:`get_started`

   global pathcing
      One of the patching options available in |intelex|.
      With global patching, you can patch all scikit-learn applications at once::

         python sklearnex.glob patch_sklearn
      
      .. seealso:: :ref:`global_patching`
   
   how it works
      We are replacing the original estimators in scikit-learn modules with ours.
      There is dispatching inside our estimators, if the desired algorithm parameters
      are not supported by the |intelex|, then the result of the original scikit-learn
      is returned.

      .. seealso:: :ref:`verbose`