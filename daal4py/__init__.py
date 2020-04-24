try:
    from _daal4py import *
    from _daal4py import __version__, __daal_link_version__, __daal_run_version__, __has_dist__
except ImportError as e:
    s = str(e)
    if 'libfabric' in s:
        raise ImportError(s + '\n\nActivating your conda environment or sourcing mpivars.[c]sh/psxevars.[c]sh may solve the issue.\n')
    else:
        raise

import logging
import warnings
import os
l = os.environ.get("IDP_SKLEARN_VERBOSE")
try:
    if not l is None:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=l.upper())
except:
    warnings.warn('Unknown level "{}" for logging.\n'
                    'Please, use one of "CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG".'.format(l))
