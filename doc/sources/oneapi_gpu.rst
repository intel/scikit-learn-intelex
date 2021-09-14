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

.. _oneapi_gpu:

##############################################################
oneAPI and GPU support in Intel(R) Extension for Scikit-learn*
##############################################################

Intel(R) Extension for Scikit-learn* supports oneAPI concepts, which
means that algorithms can be executed on different devices: CPUs and GPUs.
This is done via integration with
`dpctl <https://intelpython.github.io/dpctl/latest/index.html>`_ package that
implements core oneAPI concepts like queues and devices.

Intel(R) Extension for Scikit-learn* offers two options for running an algorithm on a
specific device with the help of dpctl:


1. Pass input data as
   `dpctl.tensor.usm_ndarray
   <https://intelpython.github.io/dpctl/latest/docfiles/dpctl.tensor_api.html#dpctl.tensor.usm_ndarray>`_
   to the algorithm. The computation will run on the device where the input data is
   located, and the result will be returned as :code:`usm_ndarray` to the same
   device.

   .. note::
     All the input data for an algorithm must reside on the same device.

   .. warning::
     The :code:`usm_ndarray` can only be consumed by the base methods
     like :code:`fit`, :code:`predict`, and :code:`transform`.
     Note that only the algorithms in Intel(R) Extension for Scikit-learn* support
     :code:`usm_ndarray`. The algorithms from the stock version of scikit-learn
     do not support this feature.
2. Use global configurations of Intel(R) Extension for Scikit-learn\*:
     1. Option :code:`target_offload` can be used to set the device primarily
        used to perform computations. Accepted values are :code:`str` and
        :code:`dpctl.SyclQueue`. If string, expected to be "auto" (the execution
        context is deduced from input data location), or SYCL* filter selector
        string. Default value is "auto"
     2. Option :code:`allow_fallback_to_host`
        is a boolean flag, that, if set, allows to fallback computation to host
        device in case when particular estimator does not support the selected
        one. Default value is :code:`False`.

These options can be set using :code:`sklearnex.set_config()` function or
:code:`sklearnex.config_context`. To obtain the current values of these options,
call :code:`sklearnex.get_config()`.

.. note::
     Functions :code:`set_config`, :code:`get_config` and :code:`config_context`
     are always patched after the :code:`sklearnex.patch_sklearn()` call.

.. rubric:: Example

An example on how to patch your code with Intel CPU/GPU optimizations:

.. code-block:: python

   from sklearnex import patch_sklearn, config_context
   patch_sklearn()

   from sklearn.cluster import DBSCAN

   X = np.array([[1., 2.], [2., 2.], [2., 3.],
               [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)
   with config_context(target_offload="gpu:0"):
      clustering = DBSCAN(eps=3, min_samples=2).fit(X)

.. warning::
     For compatibility reasons, algorithms in Intel(R) Extension for
     Scikit-learn* can be offloaded to the device with use of
     :code:`daal4py.oneapi.sycl_context`. It is recommended to use the ways
     described above for device offloading instead of using :code:`sycl_context`.


For execution on GPU, DPC++ compiler runtime and driver are required. Refer to `DPC++ system
requirements <https://software.intel.com/content/www/us/en/develop/articles/intel-oneapi-dpcpp-system-requirements.html>`_ for details.

DPC++ compiler runtime can be installed either from PyPI or Anaconda:

- Install from PyPI::

     pip install dpcpp-cpp-rt

- Install from Anaconda::

     conda install dpcpp_cpp_rt -c intel
