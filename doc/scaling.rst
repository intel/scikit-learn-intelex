###############################################
Scaling on Distributed Memory (Multiprocessing)
###############################################
It's Easy
---------
daal4py operates in SPMD-mode (Single Programm Multiple Data), which means your
program is executed on several process (e.g. simialar to MPI). Any of the
distributed algorithms return a the usable result on all processes. No worries,
you do not need mpi4py or alike for this (even though of course you are free to
use it together with mpi(4py)). Only very minimal changes are needed to your
daal4py code to allow daal4py to run on a cluster of workstations.

Initialize the distribution engine::

  daalinit()

Add the distribution parameter to the algorithm construction::

  kmi = kmeans_init(10, method="plusPlusDense", distributed=True)

When calling the actual computation each process expects an input file or input
array. Your program needs to tell each process which file/array it should
operate on. Like with other SPMD programs this is usually done conditinally on
the process id/rank ('daal4py.my_procid()'). Assume we have one file for each
process, all having the same prefix 'file' and being suffixed by a number. The
code could then look like this::

  result = kmi.compute("file{}.csv", daal4py.my_procid())

Finally stop the distribution engine::

  daalfini()

That's all for the python code::

  from daal4py import daalinit, daalfini, kmeans_init
  daalinit()
  kmi = kmeans_init(10, method="plusPlusDense", distributed=True)
  result = kmi.compute("file{}.csv", daal4py.my_procid())
  daalfini()

To actually get it exectuted on several processes use standard MPI mechanics,
like::

  mpirun -n 4 python ./kmeans.py

The binaries provided by Intel use the Intel® MPI library, but
daal4py can also be compiled for any other MPI implementation.

Supported Algorithms and Examples
---------------------------------
The following algorithms support distribution:

- PCA (pca)

  - `PCA <https://github.com/IntelPython/daal4py/blob/master/examples/pca_spmd.py>`_

- SVD (svd)

  - `SVD <https://github.com/IntelPython/daal4py/blob/master/examples/svd_spmd.py>`_

- Linear Regression Training (linear_regression_training)

  - `Linear Regression <https://github.com/IntelPython/daal4py/blob/master/examples/linear_regression_spmd.py>`_

- Ridge Regression Training (ridge_regression_training)

  - `Ridge Regression <https://github.com/IntelPython/daal4py/blob/master/examples/ridge_regression_spmd.py>`_

- Multinomial Naive Bayes Training (multinomial_naive_bayes_training)

  - `Naive Bayes <https://github.com/IntelPython/daal4py/blob/master/examples/naive_bayes_spmd.py>`_

- K-Means (kmeans_init and kmeans)

  - `K-Means <https://github.com/IntelPython/daal4py/blob/master/examples/kmeans_spmd.py>`_
