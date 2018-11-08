#!/bin/sh

for ex in *_batch.py; do
    echo "Running $ex"
    python $ex || exit
done

for ex in *_spmd.py ; do
    echo "Running $ex"
    mpirun -n 4 python $ex -- 2>&1 || exit
done
