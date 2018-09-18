.. daal4py documentation master file, created by
   sphinx-quick-start on Mon Jul  9 08:04:40 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

################################################
Fast and Scalable Machine Learning With DAAL4PY
################################################

Make your Machine Learning algorithms lightning fast.
Easily and efficiently scale them out to clusters of workstations.

It's Easy
---------
A daal4py machine learning algorithm gets constructed with a rich set of
parameters. Let's assume we want to find the initial set of centroids for kmeans.
We create an algorithm and configure it for 10 clusters using the 'PlusPlus' method::

    kmi = kmeans_init(10, method="plusPlusDense")

Assuming we have all our data in a CSV file we can now call it::

    result = kmi.compute('data.csv')

Our result will hold the computed centroids in the 'centroids' attribute::

    print(result.centroids)

The full example could look like this::

    from daal4py import kmeans_init
    result = kmeans_init(10, method="plusPlusDense").compute('data.csv')
    print(result.centroids)

You can even run this on a cluster by simple adding initializing/finalizing the
network and adding a keyword-parameter::

    from daal4py import daalinit, daalfini, kmeans_init
    daalinit()
    kmeans_init(10, method="plusPlusDense", distributed=True).compute(list_of_files)
    daalfini()

It's Fast
---------
Besides superior (e.g. close to native C++ Intel DAAL) performance on a single
node, the distribution mechanics of daal4py provides excellent strong and weak
scaling. It nicely handles distributing a fixed input size on increasing
clusters sizes
(strong scaling: orange) which addresses possible response time requirements. It also scales with
growing input size (weak scaling: yellow) which is needed if the data no longer fits
into memory of a single node.

.. figure:: d4p-linreg-scale.jpg

	    On a 32-node cluster (1280 cores) daal4py computed linear regression
	    of 2.86 TB of data in less than 2 seconds and 91.55 GB of data in
	    less than 74 milliseconds.

.. figure:: d4p-kmeans-scale.jpg

	    On a 32-node cluster (1280 cores) daal4py computed K-Means (10
	    clusters) of 1.12 TB of data in 108.8 seconds and 35.76 GB of data
	    in 5.26 seconds.


Configuration: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz, EIST/Turbo on 2
sockets, 20 cores per socket, 192 GB RAM, 16 nodes connected with Infiniband,
Oracle Linux Server release 7.4, using 64-bit floating point numbers

Getting daal4py
---------------
daal4py is distributed as source only. Sources and build instructions are
available at https://github.com/IntelPython/daal4py.

Overview
--------
All algorithms work the same way:

1. Instantiate and parameterize
2. Run/compute on input data

The below tables list the accepted arguments. Those with no default (None) are required arguments. All other arguments with defaults are optional and can be provided as keyword arguments (like ``optarg=77``).
Each algorithm returns a class-like object with properties as its result.

For algorithms with training and prediction, simply extract the ``model`` property from the result returned by the training and pass it in as the (second) input argument.

Note that all input objects and the result/model properties are native types, e.g. standard types (integer, float, numpy arrays, ...). Additionally, if you provide the name of a csv-file as an input argument daal4py will work on the entire file content.

Content
-------
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   scaling
   algorithms
