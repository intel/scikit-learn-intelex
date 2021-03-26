.. _verbose:

############################################
Intel(R) Extension for Scikit-learn* verbose
############################################

To find out which implementation of the algorithm is currently used,
set the environment variable.

On Linux and Mac OS::

    export SKLEARNEX_VERBOSE=INFO

On Windows::

    set SKLEARNEX_VERBOSE=INFO

During the calls that use Intel-optimized scikit-learn, you will receive additional print statements
that indicate which implementation is being called.
These print statements are only available for :ref:`scikit-learn algorithms with daal4py patches <sklearn_algorithms>`.

For example, for DBSCAN you get one of these print statements depending on which implementation is used::

    INFO: sklearn.cluster.DBSCAN.fit: uses Intel(R) oneAPI Data Analytics Library solver

::

    INFO: sklearn.cluster.DBSCAN.fit: uses original Scikit-learn solver
