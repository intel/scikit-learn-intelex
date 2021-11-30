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

.. _memory_requirements:

###################
Memory requirements 
###################

By default |intelex| algorithms run in multithread mode using all available threads. 
Depending on this the amount of RAM consumed by optimized algorithms may be more than the stock Scikit-learn.

SVM RAM consumption
-------------------
In single-thread mode both Scikit-learn and |intelex| consumes approximately the same amount of RAM.
In default |intelex| multithread mode with ``N`` number of threads algorithm will consume in ``N`` times more RAM.

GPU memory consumption
----------------------
In all |intelex| algorithms with GPU support computations run on device memory. 
The device memory size must be large enough to copy the entire dataset.
Additional device memory may also be required for internal arrays used in computation.
