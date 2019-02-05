try:
    from _daal4py import *
    from _daal4py import __version__, __daal_link_version__, __daal_run_version__
except ImportError as e:
    s = str(e)
    if 'libfabric' in s:
        raise ImportError(s + '\n\nActivating your conda environment or sourcing mpivars.[c]sh/psxevars.[c]sh may solve the issue.\n')
    else:
        raise