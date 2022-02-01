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

You may enable patching in different ways:

.. include:: /patching/patching-options.rst

.. rubric:: Example

.. include:: /patching/patch-kmeans-example.rst

In the example above, you can see that the use of the original Scikit-learn
has not changed. This behavior is achieved through drop-in patching.

To :term:`undo the patch <unpatching>`, run:

.. include:: /patching/undo-patch.rst

You may specify which algorithms to patch:

- Patching only one algorithm:

  .. include:: /patching/patch-one-algorithm.rst

- Patching several algorithms:

  .. include:: /patching/patch-several-algorithms.rst
