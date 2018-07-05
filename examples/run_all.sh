#!/bin/sh
for ex in *_batch.py; do
    echo "Running $ex"
    python $ex
done
for ex in *_spmd.py *_spv.py; do
    echo "Running $ex"
    mpirun -genv DIST_CNC=MPI -n 4 python $ex
done
