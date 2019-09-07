# This is the recipe for tailored build with limited set of algorithms from Intel® Data Analytics Acceleration Library (Intel® DAAL), appropriate build of DAAL is used

Dependencies for daal, daal-devel and mpi are excluded, NO_DIST=1 so distributed modes are not supported.

Location of DAAL build must be specified in $DAALROOT environment variable, in example below it is $DAALDIR.

For build use command:

DAALROOT=$DAALDIR conda-build tailored-recipe/ -c intel

