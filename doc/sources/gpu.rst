.. _gpu:

##############################################################
oneAPI and GPU support in Intel(R) Extension for Scikit-learn*
##############################################################

daal4py is installed with Intel(R) Extension for Scikit-learn* and provides support of oneAPI concepts, such as context and queues, which means that
algorithms can be executed on different devices, GPUs in particular. This is implemented via ``with sycl_context("xpu")``
blocks that redirect execution to a device of the selected type: GPU, CPU, or host.
Same approach is implemented for Intel(R) Extension for Scikit-learn*, so scikit-learn programs can be
executed on GPU devices as well.

To patch your code with Intel CPU/GPU optimizations:

.. code-block:: python

   from sklearnex import patch_sklearn
   from daal4py.oneapi import sycl_context
   patch_sklearn()

   from sklearn.cluster import DBSCAN

   X = np.array([[1., 2.], [2., 2.], [2., 3.],
               [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)
   with sycl_context("gpu"):
      clustering = DBSCAN(eps=3, min_samples=2).fit(X)

For execution on GPU, DPC++ compiler runtime and driver are required. Refer to `DPC++ system
requirements <https://software.intel.com/content/www/us/en/develop/articles/intel-oneapi-dpcpp-system-requirements.html>`_ for details.

DPC++ compiler runtime can be installed either from PyPI or Anaconda:

- Install from PyPI::

     pip install dpcpp-cpp-rt

- Install from Anaconda::

     conda install dpcpp_cpp_rt -c intel
