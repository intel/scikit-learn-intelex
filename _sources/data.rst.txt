.. _data:

##########
Input Data
##########
All array arguments to compute functions and to algorithm constructors can be
provided in different formats. daal4py will automatically do its best to work on
the provided data with minimal overhead, most notably without copying the data.

Numpy Arrays
------------
daal4py can directly handle all types of numpy arrays with numerical data
without copying the entire data. Arrays can be homogeneous (e.g. simple dtype) or
heterogeneous (structured array) as well as contiguous or non-contiguous.

Pandas DataFrames
-----------------
daal4py directly accepts pandas DataFrames with columns of numerical data. No
extra full copy is required.

SciPy Sparse CSR Matrix
-----------------------
daal4py can directly handle matrices of type scipy.sparse.csr_matrix without
copying the entire data.

Note: some algorithms can be configured to use an optimized compute path for CSR
data. It is required to explicitly specify the CSR method, otherwise the default
and less efficient method is used.

CSV Files
---------
The compute functions daal4py's algorithms additionally accept
CSV-filenames. Internally, daal4py will use DAAL's fast CSV reader to create
contiguous homogeneous tables.
