###############################################
Scaling on Distributed Memory (Multiprocessing)
###############################################
Some algorithms can work on several processes (distributed). Currently the
following algorithms support distribution:

- pca
- svd
- linear_regression_training
- multinomial_naive_bayes_training
- kmeans and kmeans_init

The provided binaries use the Intel® MPI library. 

Note: Distribution needs to be initialized with a call to daalinit() and before exiting the program a call to daalfini() is required.

Distributed Single Process View (DSPV)
--------------------------------------
The distributed-Single-Process-View (DSVP) mode lets you program as if you had a
single process. Distribution is done under the hood. This mode works like as
follows:

1. Your program begins with ``daalinit()``
2. Provide several arrays or csv-files (like one per process). This represents the partitioning of your data.
3. When calling an algorithm add the argument ``distributed=True``.

Single Program Multiple Data (SPMD)
-----------------------------------
You can also program in SPMD-style, e.g. like the usual MPI style. You do not
need mpi4py for daal4py/daal4r, but you can use it of course if required. Any of
the above algorithms return a the usable result on all processes. Note: daal4py
provides ``my_procid()`` and ``num_procs()`` to able to properly work in SPMD
mode. This mode works like as follows:

1.	Your program begins with ``daalinit(spmd=True)``
2.	Provide the same number of arrays or csv-files per process. This represents the partitioning of your data.
3.	When calling an algorithm add the argument ``distributed=True``.

Starting Processes
------------------
Currently we support the MPI used by Intel® Concurrent Collections (Intel CnC),
which is usually Intel® MPI Library. Just use mpirun as usual and set the
communication-protocol to MPI like this

``mpirun genv DIST_CNC=MPI -n <num-procs> python <your-program>``
