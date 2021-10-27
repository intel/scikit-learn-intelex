#!/bin/bash

# copy jupyter notebooks
cd ..
cp examples/notebooks/*.ipynb doc/sources/samples

# build the documentation
cd doc
make html