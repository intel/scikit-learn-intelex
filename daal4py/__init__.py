try:
    from _daal4py import *
    from _daal4py import __version__, __daal_link_version__, __daal_run_version__
except ImportError as e:
    s = str(e)
    print(s)
    if 'libfabric' in s:
        print('Activating your conda environment or sourcing mpivars.sh/psxevars.sh may solve the issue.')
