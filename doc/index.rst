.. daal4py documentation master file, created by
   sphinx-quickstart on Mon Jul  9 08:04:40 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

################################################
Fast and Scalabale Machine Learning With DAAL4PY
################################################

It's easy
---------
Wth this API your Python programs can use Intel(R) DAAL algorithms in just one
line:

``kmeans_init(data, 10, method="plusPlusDense").compute('data.csv')``

You can even run this on a cluster by simple adding a keyword-parameter

``kmeans_init(data, 10, method="plusPlusDense", distributed=TRUE).compute(list_of_files)``

**This is a technical preview, not a product. Intel might decide to discontinue this project at any time.**

Overview
--------
All algorithms work the same way:

1. Instantiate and parametrize
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
