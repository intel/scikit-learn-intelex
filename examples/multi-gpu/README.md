# Multi-GPU Examples

This folder contains examples of usage of Scikit-learn algorithm implementations capable of leveraging additional hardware resources and parallelization techniques via the Intel(R) extension for Scikit-learn.

### SPMD

Examples with "spmd" (single program, multiple data) suffix illustrate how multi-gpu systems can be leveraged with dpctl tensors and queues, mpi4py functionality for communication, and sklearnex algorithms. More detailed information can be found at the [source](https://github.com/intel/scikit-learn-intelex/tree/main/sklearnex/spmd).

### DPCTL

Examples with "dpctl" suffix demonstrate integration of sklearnex and [dpctl](https://github.com/IntelPython/dpctl), a data parallel control library.

### DPNP

Examples with "dpnp" suffix demonstrate integration of sklearnex and [dpnp](https://github.com/IntelPython/dpnp), the data parallel extension for numpy.
