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
The above described approach assume no awarenes of the distribution, it works as
a Distributed Single Process View (DSPV). It has a inherent scalability bottleneck, because
files/data always goes through a master node. Most processes in the network
only serve as 'slaves'.

To get around this bottleneck daal4py also provides a SPMD-mode, e.g. allows
programming in the usual MPI style. No worries, you do not need mpi4py or alike
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

  - `DSPV PCA <https://github.intel.com/SAT/daal4py/blob/master/examples/pca_dspv.py>`_
  - `SPMD PCA <https://github.intel.com/SAT/daal4py/blob/master/examples/pca_spmd.py>`_

- SVD (svd)

  - `DSPV SVD <https://github.intel.com/SAT/daal4py/blob/master/examples/svd_dspv.py>`_
  - `SPMD SVD <https://github.intel.com/SAT/daal4py/blob/master/examples/svd_spmd.py>`_

- Linear Regression Training (linear_regression_training)

  - `DSPV Linear Regression <https://github.intel.com/SAT/daal4py/blob/master/examples/linear_regression_dspv.py>`_
  - `SPMD Linear Regression <https://github.intel.com/SAT/daal4py/blob/master/examples/linear_regression_spmd.py>`_

- Ridge Regression Training (ridge_regression_training)

  - `DSPV Ridge Regression <https://github.intel.com/SAT/daal4py/blob/master/examples/ridge_regression_dspv.py>`_
  - `SPMD Ridge Regression <https://github.intel.com/SAT/daal4py/blob/master/examples/ridge_regression_spmd.py>`_

- Multinomial Naive Bayes Training (multinomial_naive_bayes_training)

  - `DSPV Naive Bayes <https://github.intel.com/SAT/daal4py/blob/master/examples/naive_bayes_dspv.py>`_
  - `SPMD Naive Bayes <https://github.intel.com/SAT/daal4py/blob/master/examples/naive_bayes_spmd.py>`_

- K-Means (kmeans_init and kmeans)

  - `DSPV K-Mmeans <https://github.intel.com/SAT/daal4py/blob/master/examples/kmeans_dspv.py>`_
  - `SPMD K-Means <https://github.intel.com/SAT/daal4py/blob/master/examples/kmeans_spmd.py>`_
