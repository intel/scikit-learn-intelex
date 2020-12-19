.. _streaming:

##############
Streaming Data
##############
For large quantities of data it might be impossible to provide all input data at
once. This might be because the data resides in multiple files and merging it is
to costly (or not feasible in other ways). In other cases the data is simply too
large to be loaded completely into memory. Or, the data might come in as an
actual stream. daal4py's streaming mode allows you to process such data.

Besides supporting certain use cases, streaming also allows interleaving I/O
operations with computation.

daal4py's streaming mode is as easy as follows:

1. When constructing the algorithm configure it with ``streaming=True``::

     algo = daal4py.svd(streaming=True)
2. Repeat calling ``compute(input-data)`` with chunks of your input (arrays, DataFrames or
   files)::

     for f in input_files:
         algo.compute(f)
3. When done with inputting, call ``finalize()`` to obtain the result::

     result = algo.finalize()

The streaming algorithms also accept arrays and DataFrames as input, e.g. the
data can come from a stream rather than from multiple files. Here is an example
which simulates a data stream using a generator which reads a file in chunks:
`SVD reading stream of data <https://github.com/IntelPython/daal4py/blob/master/examples/stream.py>`_

Supported Algorithms and Examples
---------------------------------
The following algorithms support streaming:

- SVD (svd)

  - `SVD <https://github.com/IntelPython/daal4py/blob/master/examples/svd_streaming.py>`_

- Linear Regression Training (linear_regression_training)

  - `Linear Regression <https://github.com/IntelPython/daal4py/blob/master/examples/linear_regression_streaming.py>`_

- Ridge Regression Training (ridge_regression_training)

  - `Ridge Regression <https://github.com/IntelPython/daal4py/blob/master/examples/ridge_regression_streaming.py>`_

- Multinomial Naive Bayes Training (multinomial_naive_bayes_training)

  - `Naive Bayes <https://github.com/IntelPython/daal4py/blob/master/examples/naive_bayes_streaming.py>`_

- Moments of Low Order

  - `Low Order Moments <https://github.com/IntelPython/daal4py/blob/master/examples/low_order_moms_streaming.py>`_
