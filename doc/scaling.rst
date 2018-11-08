###############################################
Scaling on Distributed Memory (Multiprocessing)
###############################################
It's Easy
---------
Only very minimal changes are needed to your daal4py code to allow it running on
a cluster of workstations. Initialize the distribution engine::

  daalinit()

Add the distribution parameter to the algorithm construction::

  kmi = kmeans_init(10, method="plusPlusDense", distributed=TRUE)

When calling the actual computation provide a list of input files or
input arrays. The input list represents the partitioning of your data, e.g. each
file/array will be processed on a separate process::

  result = kmi.compute([file1, file2, file3, file4])

Finally stop the distribution engine::

  daalfini()

That's all for the python code::

  from daal4py import daalinit, daalfini, kmeans_init
  daalinit()
  kmi = kmeans_init(10, method="plusPlusDense", distributed=TRUE)
  result = kmi.compute([file1, file2, file3, file4])
  daalfini()

To actually get it exectued on several processes
se standard MPI mechanics, like::

  mpirun -n 4 python ./kmeans.py

The provided binaries use the Intel® MPI library, but can also be compiled for MPICH.

Single Program Multiple Data (SPMD)
-----------------------------------

daal4py provides a SPMD-mode, e.g. allows programming in the usual MPI style.
No worries, you do not need mpi4py or alike
for this (even though of course you are free to use it together with daal4py).

Any of the above algorithms return a the usable result on all processes if the
distribution engine was initialized for SPMD::

  daalinit(spmd=True)

The only other difference to the above is that your program needs to tell each
process which file/array it should operate on. Like with other SPMD programs
this is usually done conditinally on the process id/rank. daal4py provides
``my_procid()`` and ``num_procs()`` to do exactly that::

  result = kmi.compute('input-{}.csv'.format(my_procid()))

Don't forget to configure your algorithm with ``distributed=True``::

  from daal4py import daalinit, daalfini, kmeans_init
  daalinit(spmd=True)
  kmi = kmeans_init(10, method="plusPlusDense", distributed=TRUE)
  result = kmi.compute('input-{}.csv'.format(my_procid()))
  daalfini()

Runnign SPMD programs works the same way::

  mpirun genv DIST_CNC=MPI -n <num-procs> python <your-program>

Supported Algorithms and Examples
---------------------------------
The following algorithms support distribution:

- PCA (pca)

  - `SPMD PCA <https://github.com/IntelPython/daal4py/blob/master/examples/pca_spmd.py>`_

- SVD (svd)

  - `SPMD SVD <https://github.com/IntelPython/daal4py/blob/master/examples/svd_spmd.py>`_

- Linear Regression Training (linear_regression_training)

  - `SPMD Linear Regression <https://github.com/IntelPython/daal4py/blob/master/examples/linear_regression_spmd.py>`_

- Ridge Regression Training (ridge_regression_training)

  - `SPMD Ridge Regression <https://github.com/IntelPython/daal4py/blob/master/examples/ridge_regression_spmd.py>`_

- Multinomial Naive Bayes Training (multinomial_naive_bayes_training)

  - `SPMD Naive Bayes <https://github.com/IntelPython/daal4py/blob/master/examples/naive_bayes_spmd.py>`_

- K-Means (kmeans_init and kmeans)

  - `SPMD K-Means <https://github.com/IntelPython/daal4py/blob/master/examples/kmeans_spmd.py>`_
